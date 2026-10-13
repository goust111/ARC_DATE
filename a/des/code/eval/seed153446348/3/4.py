from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
FRAME_COLOR = 9
TEMPLATE_LIBRARY = (
    (".#.", "###", ".#.", ".##"),
    ("..#..", ".###.", "..#..", ".##.."),
    ("..#..", "#####", "...#.", "..##."),
    (".#.", "###", "##.", ".#."),
    ("..#..", ".###.", "..##.", "...#.", ".###."),
    ("..#..", ".###.", ".#...", ".##.."),
)
CELL_SIZES = (5, 7)
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8)
ORIENTATION_TO_OFFSET = {
    0: lambda s: (0, s // 2),
    1: lambda s: (s // 2, s - 1),
    2: lambda s: (s - 1, s // 2),
    3: lambda s: (s // 2, 0),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _find_frame(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    best: tuple[int, int, tuple[int, int, int, int]] | None = None
    for color in sorted(set(map(int, grid.ravel())) - {BLACK}):
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if hh < 4 or ww < 4:
                continue
            border_len = 2 * (hh + ww) - 4
            if len(comp) != border_len:
                continue
            comp_set = set(comp)
            ok = True
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    on_border = rr in (r0, r1) or cc in (c0, c1)
                    if on_border != ((rr, cc) in comp_set):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            score = border_len
            cand = (score, color, (r0, c0, r1, c1))
            if best is None or cand[0] > best[0]:
                best = cand
    assert best is not None, "expected one large rectangular frame"
    return best[1], best[2]


def _parse_template(lines: tuple[str, ...]) -> np.ndarray:
    h = len(lines)
    w = len(lines[0])
    arr = np.zeros((h, w), dtype=bool)
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch == "#":
                arr[r, c] = True
    return arr


def _rotate_mask_and_anchor(
    mask: np.ndarray, anchor: tuple[int, int], clockwise_quarters: int
) -> tuple[np.ndarray, tuple[int, int]]:
    out = mask.copy()
    ar, ac = anchor
    for _ in range(clockwise_quarters % 4):
        h, _w = out.shape
        out = np.rot90(out, -1)
        ar, ac = ac, h - 1 - ar
    return out, (ar, ac)


def _template_anchor(mask: np.ndarray) -> tuple[int, int]:
    h, w = mask.shape
    mid = w // 2
    assert w % 2 == 1 and mask[0, mid], "template must expose a top-mid anchor"
    assert int(mask[0].sum()) == 1, "template top row must contain only the anchor pixel"
    assert h >= 2 and int(mask.sum()) >= 4
    return 0, mid


def _infer_cell_size(
    anchor_positions: list[tuple[int, int]], inner_h: int, inner_w: int
) -> int:
    candidates: list[int] = []
    max_size = min(inner_h, inner_w)
    for size in range(max_size, 2, -1):
        if size % 2 == 0 or inner_h % size or inner_w % size:
            continue
        mid = size // 2
        allowed = {(0, mid), (mid, size - 1), (size - 1, mid), (mid, 0)}
        if all((r % size, c % size) in allowed for r, c in anchor_positions):
            candidates.append(size)
    assert candidates, "could not infer the hidden grid pitch"
    return candidates[0]


def _extract_templates(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    needed_colors: set[int],
) -> dict[int, tuple[np.ndarray, tuple[int, int]]]:
    r0, c0, r1, c1 = frame_bbox
    outside = np.ones_like(grid, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    templates: dict[int, tuple[np.ndarray, tuple[int, int]]] = {}
    for color in sorted(needed_colors):
        comps = _components((grid == color) & outside)
        assert len(comps) == 1, "each anchor color must have exactly one outside template"
        comp = comps[0]
        tr0, tc0, tr1, tc1 = _bbox(comp)
        mask = np.zeros((tr1 - tr0 + 1, tc1 - tc0 + 1), dtype=bool)
        for rr, cc in comp:
            mask[rr - tr0, cc - tc0] = True
        templates[color] = (mask, _template_anchor(mask))
    return templates


def _paint_sprite(
    out: np.ndarray,
    inner_origin: tuple[int, int],
    inner_shape: tuple[int, int],
    color: int,
    mask: np.ndarray,
    sprite_anchor: tuple[int, int],
    target_anchor: tuple[int, int],
) -> None:
    inner_r0, inner_c0 = inner_origin
    inner_h, inner_w = inner_shape
    sr, sc = sprite_anchor
    for rr, cc in zip(*np.where(mask), strict=False):
        gr = target_anchor[0] + int(rr) - sr
        gc = target_anchor[1] + int(cc) - sc
        assert 0 <= gr < inner_h and 0 <= gc < inner_w
        out[inner_r0 + gr, inner_c0 + gc] = color


def _apply_rule_from_known_layout(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    cell_size: int,
    assignments: list[dict[str, object]],
    templates: dict[int, tuple[np.ndarray, tuple[int, int]]],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    r0, c0, r1, c1 = frame_bbox
    inner_origin = (r0 + 1, c0 + 1)
    inner_shape = (r1 - r0 - 1, c1 - c0 - 1)
    for item in sorted(assignments, key=lambda x: (int(x["anchor_abs"][0]), int(x["anchor_abs"][1]))):
        color = int(item["color"])
        direction = int(item["direction"])
        base_mask, base_anchor = templates[color]
        rot_mask, rot_anchor = _rotate_mask_and_anchor(base_mask, base_anchor, direction)
        _paint_sprite(
            out,
            inner_origin,
            inner_shape,
            color,
            rot_mask,
            rot_anchor,
            tuple(map(int, item["anchor_abs"])),
        )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match frame-internal anchor pixels to outside sprite templates, rotate by anchor position,
    and stamp the rotated sprite inside the frame.

    Input structure:
    - black background with one large single-pixel rectangular frame.
    - inside the frame, single colored anchor pixels sit at side-midpoints of an implicit square grid.
    - outside the frame, each anchor color has one connected monochrome sprite template whose default
      anchor is the single top-middle pixel of its bounding box.

    Rule:
    - infer the hidden cell size from anchor coordinates;
    - decode anchor orientation from its within-cell offset;
    - rotate the matching outside template by 0/90/180/270 clockwise;
    - stamp the rotated template into the frame with its rotated anchor aligned to the anchor pixel;
    - process anchors in reading order, so later anchors overwrite earlier overlaps.
    """
    g = np.asarray(grid, dtype=int)
    _frame_color, frame_bbox = _find_frame(g)
    r0, c0, r1, c1 = frame_bbox
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    anchor_points = [(int(r), int(c), int(inner[r, c])) for r, c in zip(*np.where(inner != BLACK), strict=False)]
    assert anchor_points, "expected anchor pixels inside the frame"
    anchor_positions = [(r, c) for r, c, _ in anchor_points]
    cell_size = _infer_cell_size(anchor_positions, inner.shape[0], inner.shape[1])
    mid = cell_size // 2
    offset_to_dir = {
        (0, mid): 0,
        (mid, cell_size - 1): 1,
        (cell_size - 1, mid): 2,
        (mid, 0): 3,
    }
    needed_colors = {color for _r, _c, color in anchor_points}
    templates = _extract_templates(g, frame_bbox, needed_colors)

    assignments: list[dict[str, object]] = []
    for r, c, color in anchor_points:
        direction = offset_to_dir[(r % cell_size, c % cell_size)]
        assignments.append(
            {
                "color": int(color),
                "direction": int(direction),
                "anchor_abs": (int(r), int(c)),
            }
        )
    return _apply_rule_from_known_layout(g, frame_bbox, cell_size, assignments, templates)


def _template_library_masks() -> list[np.ndarray]:
    return [_parse_template(lines) for lines in TEMPLATE_LIBRARY]


def _rectangles_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_outside_templates(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    template_by_color: dict[int, np.ndarray],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    fr0, fc0, fr1, fc1 = frame_bbox
    blocked = np.zeros((h, w), dtype=bool)
    blocked[max(0, fr0 - 1) : min(h, fr1 + 2), max(0, fc0 - 1) : min(w, fc1 + 2)] = True

    items = list(template_by_color.items())
    rng.shuffle(items)
    items.sort(key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True)
    for color, mask in items:
        th, tw = mask.shape
        candidates: list[tuple[int, int]] = []
        for r in range(h - th + 1):
            for c in range(w - tw + 1):
                rect = (r, c, r + th - 1, c + tw - 1)
                if _rectangles_intersect(rect, frame_bbox):
                    continue
                if blocked[r : r + th, c : c + tw].any():
                    continue
                candidates.append((r, c))
        if not candidates:
            return False
        r, c = rng.choice(candidates)
        grid[r : r + th, c : c + tw][mask] = int(color)
        blocked[max(0, r - 1) : min(h, r + th + 1), max(0, c - 1) : min(w, c + tw + 1)] = True
    return True


def _placement_options(
    rows: int,
    cols: int,
    cell_size: int,
    template_by_color: dict[int, np.ndarray],
) -> dict[tuple[int, int], list[dict[str, object]]]:
    inner_h = rows * cell_size
    inner_w = cols * cell_size
    options: dict[tuple[int, int], list[dict[str, object]]] = {}
    for cell_r in range(rows):
        for cell_c in range(cols):
            cell = (cell_r, cell_c)
            opts: list[dict[str, object]] = []
            for color, base_mask in template_by_color.items():
                base_anchor = _template_anchor(base_mask)
                for direction in range(4):
                    anchor_abs = (
                        cell_r * cell_size + ORIENTATION_TO_OFFSET[direction](cell_size)[0],
                        cell_c * cell_size + ORIENTATION_TO_OFFSET[direction](cell_size)[1],
                    )
                    rot_mask, rot_anchor = _rotate_mask_and_anchor(base_mask, base_anchor, direction)
                    cover: set[tuple[int, int]] = set()
                    ok = True
                    for rr, cc in zip(*np.where(rot_mask), strict=False):
                        gr = anchor_abs[0] + int(rr) - rot_anchor[0]
                        gc = anchor_abs[1] + int(cc) - rot_anchor[1]
                        if not (0 <= gr < inner_h and 0 <= gc < inner_w):
                            ok = False
                            break
                        cover.add((gr, gc))
                    if ok:
                        opts.append(
                            {
                                "cell": cell,
                                "color": int(color),
                                "direction": int(direction),
                                "anchor_abs": anchor_abs,
                                "cover": cover,
                            }
                        )
            options[cell] = opts
    return options


def _choose_anchor_assignments(
    rows: int,
    cols: int,
    colors: list[int],
    options: dict[tuple[int, int], list[dict[str, object]]],
    rng: random.Random,
) -> list[dict[str, object]] | None:
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    min_anchors = max(len(colors), 4)
    max_anchors = min(len(all_cells), len(colors) + 4)
    want_overlap = bool(rng.randrange(2))

    overlap_pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    flat_opts = [opt for cell in all_cells for opt in options[cell]]
    for i in range(len(flat_opts)):
        for j in range(i + 1, len(flat_opts)):
            a = flat_opts[i]
            b = flat_opts[j]
            if a["cell"] == b["cell"]:
                continue
            if a["cover"] & b["cover"]:
                overlap_pairs.append((a, b))
    if want_overlap and not overlap_pairs:
        return None

    for _ in range(300):
        chosen: dict[tuple[int, int], dict[str, object]] = {}
        used_coords: set[tuple[int, int]] = set()
        if want_overlap:
            seed_a, seed_b = rng.choice(overlap_pairs)
            chosen[tuple(seed_a["cell"])] = seed_a
            chosen[tuple(seed_b["cell"])] = seed_b
            used_coords |= set(seed_a["cover"])
            for cell in seed_b["cover"]:
                used_coords.add(cell)

        anchor_count = rng.randint(max(min_anchors, len(chosen)), max_anchors)
        remaining_cells = [cell for cell in all_cells if cell not in chosen]
        rng.shuffle(remaining_cells)
        chosen_cells = list(chosen) + remaining_cells[: anchor_count - len(chosen)]
        if len({r for r, _c in chosen_cells}) < 2 or len({c for _r, c in chosen_cells}) < 2:
            continue

        assigned_colors = {int(opt["color"]) for opt in chosen.values()}
        free_cells = [cell for cell in chosen_cells if cell not in chosen]
        ok = True
        for color in colors:
            if color in assigned_colors:
                continue
            candidate_cells = [cell for cell in free_cells if any(int(opt["color"]) == color for opt in options[cell])]
            if not candidate_cells:
                ok = False
                break
            cell = rng.choice(candidate_cells)
            opts = [opt for opt in options[cell] if int(opt["color"]) == color]
            if not want_overlap:
                safe = [opt for opt in opts if not (used_coords & set(opt["cover"]))]
                opts = safe or opts
            opt = rng.choice(opts)
            chosen[cell] = opt
            if not want_overlap and used_coords & set(opt["cover"]):
                ok = False
                break
            used_coords |= set(opt["cover"])
            free_cells.remove(cell)
            assigned_colors.add(color)
        if not ok:
            continue

        rng.shuffle(free_cells)
        for cell in free_cells:
            opts = list(options[cell])
            if not want_overlap:
                opts = [opt for opt in opts if not (used_coords & set(opt["cover"]))]
            if not opts:
                ok = False
                break
            opt = rng.choice(opts)
            chosen[cell] = opt
            used_coords |= set(opt["cover"])
        if not ok:
            continue

        assignments = list(chosen.values())
        counts: dict[tuple[int, int], int] = {}
        for item in assignments:
            for coord in item["cover"]:
                counts[coord] = counts.get(coord, 0) + 1
        has_overlap = any(v > 1 for v in counts.values())
        if has_overlap != want_overlap:
            continue
        return assignments
    return None


def generate_input() -> np.ndarray:
    template_masks = _template_library_masks()
    for _ in range(500):
        rng = random.Random(random.randrange(1 << 30))
        cell_size = rng.choice(CELL_SIZES)
        rows = rng.randint(2, 4)
        cols = rng.randint(2, 4)
        frame_h = rows * cell_size + 2
        frame_w = cols * cell_size + 2

        color_count = rng.randint(2, min(4, rows * cols))
        colors = rng.sample(list(COLOR_POOL), color_count)
        chosen_shapes = rng.sample(template_masks, color_count)
        template_by_color = {color: shape.copy() for color, shape in zip(colors, chosen_shapes, strict=False)}

        max_th = max(mask.shape[0] for mask in template_by_color.values())
        max_tw = max(mask.shape[1] for mask in template_by_color.values())
        top_margin = rng.randint(max_th + 2, max_th + 7)
        bottom_margin = rng.randint(max_th + 2, max_th + 7)
        left_margin = rng.randint(max_tw + 2, max_tw + 7)
        right_margin = rng.randint(max_tw + 2, max_tw + 7)

        h = frame_h + top_margin + bottom_margin
        w = frame_w + left_margin + right_margin
        frame_bbox = (
            top_margin,
            left_margin,
            top_margin + frame_h - 1,
            left_margin + frame_w - 1,
        )

        grid = np.zeros((h, w), dtype=int)
        fr0, fc0, fr1, fc1 = frame_bbox
        grid[fr0, fc0 : fc1 + 1] = FRAME_COLOR
        grid[fr1, fc0 : fc1 + 1] = FRAME_COLOR
        grid[fr0 : fr1 + 1, fc0] = FRAME_COLOR
        grid[fr0 : fr1 + 1, fc1] = FRAME_COLOR

        if not _place_outside_templates(grid, frame_bbox, template_by_color, rng):
            continue

        options = _placement_options(rows, cols, cell_size, template_by_color)
        if any(not opts for opts in options.values()):
            continue
        assignments = _choose_anchor_assignments(rows, cols, colors, options, rng)
        if assignments is None:
            continue

        for item in assignments:
            rr, cc = map(int, item["anchor_abs"])
            grid[fr0 + 1 + rr, fc0 + 1 + cc] = int(item["color"])

        expected = _apply_rule_from_known_layout(grid, frame_bbox, cell_size, assignments, {c: (m, _template_anchor(m)) for c, m in template_by_color.items()})
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
