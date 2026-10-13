import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME_COLOR = 5

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
OBJECT_COUNT_CHOICES = (2, 3)
INNER_H_RANGE = (4, 7)
INNER_W_RANGE = (4, 7)
GEN_RETRIES = 300
PLACE_RETRIES = 200
TEMPLATE_RETRIES = 80

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _corners(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    return [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]


def _rotate_pos(shape: tuple[int, int], pos: tuple[int, int], k: int) -> tuple[int, int]:
    h, w = shape
    r, c = pos
    for _ in range(k % 4):
        r, c = w - 1 - c, r
        h, w = w, h
    return int(r), int(c)


def _find_preimage_corner(
    raw_shape: tuple[int, int], target_pos: tuple[int, int], k: int
) -> tuple[int, int]:
    for cand in _corners(raw_shape):
        if _rotate_pos(raw_shape, cand, k) == target_pos:
            return cand
    raise ValueError("target corner has no preimage")


def _collect_frames_of_color(
    grid: np.ndarray, bg: int, frame_color: int
) -> list[dict]:
    frames: list[dict] = []
    for comp in _components_mask(grid == frame_color):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        hh, ww = sub.shape
        border = np.zeros((hh, ww), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(sub[border] == frame_color):
            continue
        if np.any(sub[1:-1, 1:-1] == frame_color):
            continue

        inner = sub[1:-1, 1:-1]
        inner_h, inner_w = inner.shape
        anchor_pts = np.argwhere((inner != bg) & (inner != frame_color))
        if len(anchor_pts) != 2:
            continue
        valid_corners = {
            (0, 0),
            (0, inner_w - 1),
            (inner_h - 1, 0),
            (inner_h - 1, inner_w - 1),
        }
        anchors: dict[int, tuple[int, int]] = {}
        ok = True
        for ar, ac in anchor_pts:
            pos = (int(ar), int(ac))
            color = int(inner[ar, ac])
            if pos not in valid_corners or color in anchors:
                ok = False
                break
            anchors[color] = pos
        if not ok:
            continue

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_shape": (inner_h, inner_w),
                "anchors": anchors,
                "key": frozenset(anchors),
            }
        )
    return frames


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    colors = [FRAME_COLOR] + [
        int(c) for c in np.unique(grid) if int(c) not in (bg, FRAME_COLOR)
    ]
    best: list[dict] = []
    for color in colors:
        frames = _collect_frames_of_color(grid, bg, color)
        if color == FRAME_COLOR and frames:
            return frames
        if len(frames) > len(best):
            best = frames
    return best


def _frame_cover_mask(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    cover = np.zeros(shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        cover[r0 : r1 + 1, c0 : c1 + 1] = True
    return cover


def _extract_templates(
    grid: np.ndarray, bg: int, cover: np.ndarray
) -> list[dict]:
    mask = (grid != bg) & (~cover)
    templates: list[dict] = []
    for comp in _components_mask(mask):
        r0, c0, r1, c1 = _bbox(comp)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        colors: set[int] = set()
        for r, c in comp:
            color = int(grid[r, c])
            patch[r - r0, c - c0] = color
            colors.add(color)
        templates.append(
            {
                "coords": comp,
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "colors": colors,
            }
        )
    return templates


def _align_template(
    patch: np.ndarray, color_a: int, color_b: int, frame: dict
) -> np.ndarray:
    target_shape = tuple(int(x) for x in frame["inner_shape"])
    target_a = tuple(int(x) for x in frame["anchors"][color_a])
    target_b = tuple(int(x) for x in frame["anchors"][color_b])
    for k in range(4):
        rot = np.rot90(patch, k)
        if rot.shape != target_shape:
            continue
        pos_a = np.argwhere(rot == color_a)
        pos_b = np.argwhere(rot == color_b)
        if len(pos_a) != 1 or len(pos_b) != 1:
            continue
        if tuple(map(int, pos_a[0])) == target_a and tuple(map(int, pos_b[0])) == target_b:
            return rot
    raise AssertionError("template rotation did not match frame anchors")


def _anchor_neighbor_color(
    patch: np.ndarray,
    pos: tuple[int, int],
    anchor_colors: set[int],
    bg: int,
) -> int:
    h, w = patch.shape
    colors = set()
    for nr, nc in _neighbors4(pos[0], pos[1], h, w):
        color = int(patch[nr, nc])
        if color != bg and color not in anchor_colors:
            colors.add(color)
    if len(colors) != 1:
        raise AssertionError("anchor must touch exactly one paint color")
    return int(next(iter(colors)))


def _recolor_rotated_template(
    rotated: np.ndarray, color_a: int, color_b: int, bg: int
) -> np.ndarray:
    pos_a = tuple(map(int, np.argwhere(rotated == color_a)[0]))
    pos_b = tuple(map(int, np.argwhere(rotated == color_b)[0]))
    anchor_colors = {int(color_a), int(color_b)}
    paint_a = _anchor_neighbor_color(rotated, pos_a, anchor_colors, bg)
    paint_b = _anchor_neighbor_color(rotated, pos_b, anchor_colors, bg)
    if paint_a == paint_b:
        raise AssertionError("paint colors must stay distinct")

    extra = set(map(int, np.unique(rotated))) - {bg, color_a, color_b, paint_a, paint_b}
    if extra:
        raise AssertionError("unexpected extra colors in template")

    out = rotated.copy()
    out[rotated == paint_a] = color_a
    out[rotated == paint_b] = color_b
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each template to the frame with the same anchor-color pair, rotate it so the
    template anchors coincide with the frame anchors, then recolor the template's two
    paint colors from those anchors and place it inside the frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    frames = _find_frames(g, bg)
    assert frames, "main: expected at least one hollow frame"

    cover = _frame_cover_mask(g.shape, frames)
    templates = _extract_templates(g, bg, cover)
    assert templates, "main: expected at least one external template"

    matched: dict[frozenset[int], dict] = {}
    for frame in frames:
        key = frame["key"]
        candidates = [tmpl for tmpl in templates if key.issubset(tmpl["colors"])]
        assert len(candidates) == 1, "main: template matching by anchor-color pair must be unique"
        matched[key] = candidates[0]

    out = g.copy()
    for tmpl in templates:
        for r, c in tmpl["coords"]:
            out[r, c] = bg

    for frame in frames:
        colors = list(frame["anchors"])
        color_a, color_b = int(colors[0]), int(colors[1])
        rotated = _align_template(matched[frame["key"]]["patch"], color_a, color_b, frame)
        recolored = _recolor_rotated_template(rotated, color_a, color_b, bg)

        r0, c0, _, _ = frame["bbox"]
        inner_r0, inner_c0 = r0 + 1, c0 + 1
        mask = recolored != bg
        out[inner_r0 : inner_r0 + recolored.shape[0], inner_c0 : inner_c0 + recolored.shape[1]][
            mask
        ] = recolored[mask]

    return out


def _random_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    steps = ["r"] * abs(tr - r) + ["c"] * abs(tc - c)
    random.shuffle(steps)
    dr = 1 if tr > r else -1
    dc = 1 if tc > c else -1
    path = [(r, c)]
    for step in steps:
        if step == "r":
            r += dr
        else:
            c += dc
        path.append((r, c))
    return path


def _mask_distances(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q = deque([start])
    dist[start] = 0
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _adjacent_non_anchor_colors(
    patch: np.ndarray, pos: tuple[int, int], anchor_colors: set[int]
) -> set[int]:
    h, w = patch.shape
    colors = set()
    for nr, nc in _neighbors4(pos[0], pos[1], h, w):
        color = int(patch[nr, nc])
        if color != BG and color not in anchor_colors:
            colors.add(color)
    return colors


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _dedupe_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for point in points:
        if point not in seen:
            seen.add(point)
            out.append(point)
    return out


def _make_template_patch(
    raw_h: int,
    raw_w: int,
    pos_a: tuple[int, int],
    pos_b: tuple[int, int],
    anchor_a: int,
    anchor_b: int,
    paint_a: int,
    paint_b: int,
) -> np.ndarray | None:
    for _ in range(TEMPLATE_RETRIES):
        required = [pos_a, pos_b]
        if not any(r == 0 for r, _ in required):
            required.append((0, random.randrange(raw_w)))
        if not any(r == raw_h - 1 for r, _ in required):
            required.append((raw_h - 1, random.randrange(raw_w)))
        if not any(c == 0 for _, c in required):
            required.append((random.randrange(raw_h), 0))
        if not any(c == raw_w - 1 for _, c in required):
            required.append((random.randrange(raw_h), raw_w - 1))
        required = _dedupe_points(required)

        middle = [p for p in required if p not in (pos_a, pos_b)]
        random.shuffle(middle)
        ordered = [pos_a] + middle + [pos_b]

        mask = np.zeros((raw_h, raw_w), dtype=bool)
        for start, end in zip(ordered, ordered[1:]):
            for r, c in _random_path(start, end):
                mask[r, c] = True

        extra_steps = random.randint(max(1, (raw_h + raw_w) // 3), raw_h + raw_w)
        occupied = [tuple(map(int, p)) for p in np.argwhere(mask)]
        for _ in range(extra_steps):
            sr, sc = random.choice(occupied)
            dr, dc = random.choice(DIRS4)
            nr, nc = sr + dr, sc + dc
            if 0 <= nr < raw_h and 0 <= nc < raw_w:
                mask[nr, nc] = True
                occupied.append((nr, nc))

        if not _touches_all_sides(mask):
            continue

        dist_a = _mask_distances(mask, pos_a)
        dist_b = _mask_distances(mask, pos_b)
        if np.any(mask & (dist_a < 0)) or np.any(mask & (dist_b < 0)):
            continue

        patch = np.full((raw_h, raw_w), BG, dtype=int)
        for r, c in zip(*np.where(mask)):
            rr, cc = int(r), int(c)
            if (rr, cc) == pos_a:
                patch[rr, cc] = anchor_a
            elif (rr, cc) == pos_b:
                patch[rr, cc] = anchor_b
            elif dist_a[rr, cc] <= dist_b[rr, cc]:
                patch[rr, cc] = paint_a
            else:
                patch[rr, cc] = paint_b

        if not np.any(patch == paint_a) or not np.any(patch == paint_b):
            continue
        if _adjacent_non_anchor_colors(patch, pos_a, {anchor_a, anchor_b}) != {paint_a}:
            continue
        if _adjacent_non_anchor_colors(patch, pos_b, {anchor_a, anchor_b}) != {paint_b}:
            continue
        return patch
    return None


def _can_place(occ: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = occ.shape
    if top < 0 or left < 0 or top + h > hh or left + w > ww:
        return False
    r0 = max(0, top - pad)
    r1 = min(hh, top + h + pad)
    c0 = max(0, left - pad)
    c1 = min(ww, left + w + pad)
    return not occ[r0:r1, c0:c1].any()


def _place_rectangles(rects: list[dict], h: int, w: int) -> bool:
    occ = np.zeros((h, w), dtype=bool)
    order = rects[:]
    random.shuffle(order)
    order.sort(key=lambda item: int(item["h"]) * int(item["w"]), reverse=True)
    for rect in order:
        rh, rw = int(rect["h"]), int(rect["w"])
        placed = False
        for _ in range(PLACE_RETRIES):
            top = random.randint(0, h - rh)
            left = random.randint(0, w - rw)
            if _can_place(occ, top, left, rh, rw, pad=1):
                rect["top"] = top
                rect["left"] = left
                occ[top : top + rh, left : left + rw] = True
                placed = True
                break
        if not placed:
            return False
    return True


def _render_input(canvas_h: int, canvas_w: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((canvas_h, canvas_w), BG, dtype=int)
    for spec in specs:
        fr, fc = int(spec["frame_top"]), int(spec["frame_left"])
        inner_h, inner_w = spec["inner_shape"]
        grid[fr, fc : fc + inner_w + 2] = FRAME_COLOR
        grid[fr + inner_h + 1, fc : fc + inner_w + 2] = FRAME_COLOR
        grid[fr : fr + inner_h + 2, fc] = FRAME_COLOR
        grid[fr : fr + inner_h + 2, fc + inner_w + 1] = FRAME_COLOR
        for color, (ar, ac) in spec["frame_anchors"].items():
            grid[fr + 1 + ar, fc + 1 + ac] = int(color)

        tr, tc = int(spec["template_top"]), int(spec["template_left"])
        patch = spec["template_patch"]
        mask = patch != BG
        grid[tr : tr + patch.shape[0], tc : tc + patch.shape[1]][mask] = patch[mask]
    return grid


def _render_expected(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        tr, tc = int(spec["template_top"]), int(spec["template_left"])
        patch = spec["template_patch"]
        mask = patch != BG
        out[tr : tr + patch.shape[0], tc : tc + patch.shape[1]][mask] = BG

    for spec in specs:
        transformed = np.rot90(spec["template_patch"], int(spec["rotation"]))
        recolored = transformed.copy()
        recolored[transformed == int(spec["paint_colors"][0])] = int(spec["anchor_colors"][0])
        recolored[transformed == int(spec["paint_colors"][1])] = int(spec["anchor_colors"][1])

        fr, fc = int(spec["frame_top"]), int(spec["frame_left"])
        inner = out[fr + 1 : fr + 1 + transformed.shape[0], fc + 1 : fc + 1 + transformed.shape[1]]
        mask = recolored != BG
        inner[mask] = recolored[mask]
    return out


def _gen_attempt() -> np.ndarray | None:
    canvas_h = random.randint(*GRID_H_RANGE)
    canvas_w = random.randint(*GRID_W_RANGE)
    count = random.choice(OBJECT_COUNT_CHOICES)

    usable_colors = [1, 2, 3, 4, 6, 7, 8, 9]
    paint_colors = random.sample(usable_colors, 2)
    anchor_pool = [c for c in usable_colors if c not in paint_colors]
    if len(anchor_pool) < 2 * count:
        return None
    chosen_anchor_colors = random.sample(anchor_pool, 2 * count)

    specs: list[dict] = []
    rects: list[dict] = []

    for i in range(count):
        anchor_a = int(chosen_anchor_colors[2 * i])
        anchor_b = int(chosen_anchor_colors[2 * i + 1])
        rotation = random.randrange(4)

        inner_h = random.randint(*INNER_H_RANGE)
        inner_w = random.randint(*INNER_W_RANGE)
        if rotation % 2 == 1 and random.random() < 0.7:
            for _ in range(10):
                if inner_h != inner_w:
                    break
                inner_h = random.randint(*INNER_H_RANGE)
                inner_w = random.randint(*INNER_W_RANGE)

        frame_shape = (inner_h, inner_w)
        frame_corners = _corners(frame_shape)
        idx_a, idx_b = random.sample(range(4), 2)
        frame_pos_a = frame_corners[idx_a]
        frame_pos_b = frame_corners[idx_b]

        raw_shape = frame_shape if rotation % 2 == 0 else (inner_w, inner_h)
        raw_pos_a = _find_preimage_corner(raw_shape, frame_pos_a, rotation)
        raw_pos_b = _find_preimage_corner(raw_shape, frame_pos_b, rotation)

        patch = _make_template_patch(
            raw_shape[0],
            raw_shape[1],
            raw_pos_a,
            raw_pos_b,
            anchor_a,
            anchor_b,
            int(paint_colors[0]),
            int(paint_colors[1]),
        )
        if patch is None:
            return None

        spec = {
            "inner_shape": frame_shape,
            "frame_anchors": {anchor_a: frame_pos_a, anchor_b: frame_pos_b},
            "anchor_colors": (anchor_a, anchor_b),
            "rotation": rotation,
            "template_patch": patch,
            "paint_colors": tuple(map(int, paint_colors)),
        }
        specs.append(spec)
        rects.append({"kind": "frame", "index": i, "h": inner_h + 2, "w": inner_w + 2})
        rects.append({"kind": "template", "index": i, "h": raw_shape[0], "w": raw_shape[1]})

    if not _place_rectangles(rects, canvas_h, canvas_w):
        return None

    for rect in rects:
        spec = specs[int(rect["index"])]
        if rect["kind"] == "frame":
            spec["frame_top"] = int(rect["top"])
            spec["frame_left"] = int(rect["left"])
        else:
            spec["template_top"] = int(rect["top"])
            spec["template_left"] = int(rect["left"])

    grid = _render_input(canvas_h, canvas_w, specs)
    expected = _render_expected(grid, specs)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    frames = _find_frames(grid, BG)
    if len(frames) != count:
        return None
    frame_keys = Counter(frame["key"] for frame in frames)
    if any(v != 1 for v in frame_keys.values()):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
