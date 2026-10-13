from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SEPARATOR_COLOR = 5
GEN_RETRIES = 300
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_separator_row(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    candidates = [
        r
        for r in range(1, h - 1)
        if int(g[r, 0]) != BG and bool(np.all(g[r] == g[r, 0]))
    ]
    assert len(candidates) == 1, "expected one solid separator row"
    row = int(candidates[0])
    return row, int(g[row, 0])


def _is_frame_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros_like(local)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(local, border))


def _extract_templates(upper: np.ndarray, lower: np.ndarray, separator_color: int) -> dict[int, np.ndarray]:
    needed_colors = {int(v) for v in np.unique(lower) if int(v) != BG}
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in {int(v) for v in np.unique(upper) if int(v) not in (BG, separator_color)}:
        for comp in components_mask(upper == color):
            if _is_frame_component(comp):
                candidates.append((int(color), bbox_coords(comp)))

    # Keep only outermost rings; this avoids treating any accidental inner sub-patterns as frames.
    candidates.sort(key=lambda item: -((item[1][2] - item[1][0] + 1) * (item[1][3] - item[1][1] + 1)))
    outer: list[tuple[int, tuple[int, int, int, int]]] = []
    for color, bbox in candidates:
        r0, c0, r1, c1 = bbox
        if any(r0 > or0 and c0 > oc0 and r1 < or1 and c1 < oc1 for _oc, (or0, oc0, or1, oc1) in outer):
            continue
        outer.append((color, bbox))

    templates: dict[int, np.ndarray] = {}
    for color, (r0, c0, r1, c1) in outer:
        if color not in needed_colors:
            continue
        patch = upper[r0 + 1 : r1, c0 + 1 : c1].copy()
        assert patch.size and np.any(patch != BG), "template interior must contain a visible pattern"
        assert color not in templates, "frame colors must be unique"
        templates[color] = patch

    assert templates and set(templates) == needed_colors, "every control color must map to one template"
    return templates


def _render_stamps(controls: np.ndarray, templates: dict[int, np.ndarray]) -> np.ndarray:
    lower = np.asarray(controls, dtype=int)
    out = np.zeros_like(lower)
    ordered = [tuple(map(int, rc)) for rc in np.argwhere(lower != BG)]
    ordered.sort(key=lambda rc: (rc[0], rc[1]))
    h, w = out.shape

    for r, c in ordered:
        color = int(lower[r, c])
        assert color in templates, "unknown control color"
        patch = templates[color]
        for pr, pc in np.argwhere(patch != BG):
            rr = r + int(pr)
            cc = c + int(pc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(patch[pr, pc])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combine colored stamp templates according to matching control points below a separator row.

    Input structure:
    - Black background.
    - The grid is split by one solid horizontal separator row.
    - Above the separator: several stamp templates, each surrounded by a one-pixel colored rectangle.
    - Below the separator: singleton control points whose colors match template-frame colors.

    Rule:
    - Extract each template from inside its colored frame.
    - Crop the output to the lower region only.
    - For each control point, place the matching template with the control point as the template's top-left.
    - Black template pixels are transparent; non-black pixels overwrite earlier results.
    - Controls are processed in row-major order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_row, sep_color = _find_separator_row(g)
    upper = g[:sep_row]
    lower = g[sep_row + 1 :]
    assert upper.size and lower.size, "both regions must be non-empty"

    templates = _extract_templates(upper, lower, sep_color)
    return _render_stamps(lower, templates)


def _blob_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    target = rng.randint(max(1, (h * w) // 3), h * w)
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    frontier = {start}
    while len(cells) < target and frontier:
        cr, cc = rng.choice(tuple(frontier))
        nbrs = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                nbrs.append((nr, nc))
        if nbrs:
            nxt = rng.choice(nbrs)
            cells.add(nxt)
            frontier.add(nxt)
        else:
            frontier.remove((cr, cc))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_patch(rng: random.Random, h: int, w: int, color: int) -> np.ndarray:
    mode = rng.choice(("solid", "hstripe", "vstripe", "diag", "plus", "corner", "blob"))
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]

    if mode == "solid":
        mask = np.ones((h, w), dtype=bool)
    elif mode == "hstripe":
        mask = np.broadcast_to((rows % 2) == rng.randrange(2), (h, w)).copy()
    elif mode == "vstripe":
        mask = np.broadcast_to((cols % 2) == rng.randrange(2), (h, w)).copy()
    elif mode == "diag":
        mask = ((rows + cols) % 2) == rng.randrange(2)
    elif mode == "plus":
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 2, :] = True
        mask[:, w // 2] = True
    elif mode == "corner":
        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = True
        mask[:, 0] = True
    else:
        mask = _blob_mask(rng, h, w)

    if not np.any(mask):
        mask[rng.randrange(h), rng.randrange(w)] = True
    if rng.random() < 0.5:
        mask = np.flipud(mask)
    if rng.random() < 0.5:
        mask = np.fliplr(mask)

    patch = np.zeros((h, w), dtype=int)
    patch[mask] = color
    return patch


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return not (
        ar + ah + gap <= br
        or br + bh + gap <= ar
        or ac + aw + gap <= bc
        or bc + bw + gap <= ac
    )


def _place_boxes(
    rng: random.Random,
    area_h: int,
    area_w: int,
    boxes: list[tuple[int, int]],
    gap: int = 1,
    margin: int = 1,
) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int] | None] = [None] * len(boxes)
    placed: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0] * boxes[i][1], reverse=True)
    for idx in order:
        bh, bw = boxes[idx]
        if area_h - bh < 2 * margin or area_w - bw < 2 * margin:
            return None
        for _ in range(200):
            r = rng.randint(margin, area_h - bh - margin)
            c = rng.randint(margin, area_w - bw - margin)
            box = (r, c, bh, bw)
            if any(_boxes_overlap(box, other, gap=gap) for other in placed):
                continue
            placed.append(box)
            positions[idx] = (r, c)
            break
        else:
            return None
    return [pos for pos in positions if pos is not None]


def _nonzero_cells(top: tuple[int, int], patch: np.ndarray) -> set[tuple[int, int]]:
    tr, tc = top
    return {
        (tr + int(r), tc + int(c))
        for r, c in np.argwhere(patch != BG)
    }


def _find_overlap_position(
    rng: random.Random,
    area_h: int,
    area_w: int,
    base_pos: tuple[int, int],
    base_patch: np.ndarray,
    new_patch: np.ndarray,
    used_controls: set[tuple[int, int]],
) -> tuple[int, int] | None:
    base_cells = [tuple(map(int, rc)) for rc in np.argwhere(base_patch != BG)]
    new_cells = [tuple(map(int, rc)) for rc in np.argwhere(new_patch != BG)]
    rng.shuffle(base_cells)
    rng.shuffle(new_cells)
    nh, nw = new_patch.shape

    for br, bc in base_cells:
        for nr, nc in new_cells:
            top = (base_pos[0] + br - nr, base_pos[1] + bc - nc)
            tr, tc = top
            if top in used_controls or top == base_pos:
                continue
            if 0 <= tr <= area_h - nh and 0 <= tc <= area_w - nw:
                return top
    return None


def _visible_overlap(controls: list[tuple[int, int, int]], templates: dict[int, np.ndarray]) -> bool:
    seen: dict[tuple[int, int], int] = {}
    for r, c, color in sorted(controls, key=lambda item: (item[0], item[1])):
        for cell in _nonzero_cells((r, c), templates[color]):
            if cell in seen and seen[cell] != color:
                return True
            seen[cell] = color
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_templates = rng.randint(2, 4)
    color_pool = list(rng.sample(PALETTE, 2 * n_templates))
    border_colors = color_pool[:n_templates]
    fill_colors = color_pool[n_templates:]

    patches = []
    frame_boxes = []
    for fill in fill_colors:
        ph = rng.randint(2, 5)
        pw = rng.randint(2, 5)
        patch = _make_patch(rng, ph, pw, fill)
        patches.append(patch)
        frame_boxes.append((ph + 2, pw + 2))

    min_width = max(16, max(max(w for _h, w in frame_boxes), max(p.shape[1] for p in patches)) + 4)
    width = rng.randint(min_width, 26)
    top_h = rng.randint(max(7, max(h for h, _w in frame_boxes) + 2), 15)
    bottom_h = rng.randint(max(8, max(p.shape[0] for p in patches) + 2), 15)

    frame_positions = _place_boxes(rng, top_h, width, frame_boxes, gap=1, margin=1)
    if frame_positions is None:
        return None

    controls: list[tuple[int, int, int]] = []
    base_boxes = [(p.shape[0], p.shape[1]) for p in patches]
    control_positions = _place_boxes(rng, bottom_h, width, base_boxes, gap=1, margin=0)
    if control_positions is None:
        return None

    for idx, (r, c) in enumerate(control_positions):
        controls.append((r, c, border_colors[idx]))

    overlap_mode = rng.random() < 0.5
    duplicate_mode = rng.random() < 0.5
    used_control_points = {(r, c) for r, c, _color in controls}

    if overlap_mode:
        first, second = rng.sample(range(n_templates), 2)
        overlap_pos = _find_overlap_position(
            rng,
            bottom_h,
            width,
            control_positions[first],
            patches[first],
            patches[second],
            used_control_points,
        )
        if overlap_pos is None:
            return None
        controls.append((overlap_pos[0], overlap_pos[1], border_colors[second]))
        used_control_points.add(overlap_pos)

    extra_count = rng.randint(0, 1)
    if duplicate_mode:
        extra_count += 1

    boxes_now = [(r, c, patches[border_colors.index(color)].shape[0], patches[border_colors.index(color)].shape[1]) for r, c, color in controls]
    for extra_idx in range(extra_count):
        if duplicate_mode and extra_idx == 0:
            color_idx = rng.randrange(n_templates)
        else:
            color_idx = rng.randrange(n_templates)
        ph, pw = patches[color_idx].shape
        for _ in range(120):
            r = rng.randint(0, bottom_h - ph)
            c = rng.randint(0, width - pw)
            if (r, c) in used_control_points:
                continue
            box = (r, c, ph, pw)
            if any(_boxes_overlap(box, other, gap=0) for other in boxes_now):
                continue
            controls.append((r, c, border_colors[color_idx]))
            used_control_points.add((r, c))
            boxes_now.append(box)
            break
        else:
            if duplicate_mode and extra_idx == 0:
                return None

    templates = {border_colors[i]: patches[i] for i in range(n_templates)}
    if overlap_mode and not _visible_overlap(controls, templates):
        return None

    grid = np.zeros((top_h + 1 + bottom_h, width), dtype=int)
    grid[top_h, :] = SEPARATOR_COLOR

    for idx, (fr, fc) in enumerate(frame_positions):
        patch = patches[idx]
        border = border_colors[idx]
        fh, fw = frame_boxes[idx]
        grid[fr, fc : fc + fw] = border
        grid[fr + fh - 1, fc : fc + fw] = border
        grid[fr : fr + fh, fc] = border
        grid[fr : fr + fh, fc + fw - 1] = border
        grid[fr + 1 : fr + fh - 1, fc + 1 : fc + fw - 1] = patch

    for r, c, color in controls:
        grid[top_h + 1 + r, c] = color

    # The solver should recover exactly the intended lower canvas.
    out = main(grid)
    if out.shape != (bottom_h, width):
        return None
    if not np.any(out != BG):
        return None
    if overlap_mode and not _visible_overlap(controls, templates):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-composition input")
