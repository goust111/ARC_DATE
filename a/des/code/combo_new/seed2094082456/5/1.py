import itertools
import random

import numpy as np


BG = 0
FRAME_COLOR = 5
GEN_RETRIES = 400
TEMPLATE_SIDE_CHOICES = (3, 4)
SCALE_CHOICES = (2, 3)
TEMPLATE_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _extract_frames(grid: np.ndarray, bg: int) -> list[dict]:
    candidates: dict[tuple[int, int, int], list[dict]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for pts in _components(grid == color):
            r0, c0, r1, c1 = _bbox(pts)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            if len(pts) != 2 * h + 2 * w - 4:
                continue
            sub = np.zeros((h, w), dtype=bool)
            sub[pts[:, 0] - r0, pts[:, 1] - c0] = True
            border = np.zeros_like(sub)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.array_equal(sub, border):
                continue
            item = {
                "color": color,
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "inner_h": h - 2,
                "inner_w": w - 2,
            }
            candidates.setdefault((color, h - 2, w - 2), []).append(item)
    assert candidates, "expected at least one rectangular frame"
    best_key = max(candidates, key=lambda k: len(candidates[k]))
    frames = sorted(candidates[best_key], key=lambda d: (d["r0"], d["c0"]))
    return frames


def _extract_templates(grid: np.ndarray, bg: int, frames: list[dict]) -> tuple[list[np.ndarray], dict[int, int]]:
    inside_frame = np.zeros_like(grid, dtype=bool)
    for fr in frames:
        inside_frame[fr["r0"] + 1 : fr["r1"], fr["c0"] + 1 : fr["c1"]] = True

    template_mask = (grid != bg) & (~inside_frame)
    for fr in frames:
        template_mask[fr["r0"] : fr["r1"] + 1, fr["c0"] : fr["c1"] + 1] &= grid[
            fr["r0"] : fr["r1"] + 1, fr["c0"] : fr["c1"] + 1
        ] != fr["color"]

    templates: list[np.ndarray] = []
    color_to_template: dict[int, int] = {}
    for pts in _components(template_mask):
        r0, c0, r1, c1 = _bbox(pts)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        idx = len(templates)
        templates.append(patch)
        for color in np.unique(patch):
            color = int(color)
            if color == bg:
                continue
            assert color not in color_to_template, "template colors must uniquely identify a template"
            color_to_template[color] = idx

    assert templates, "expected at least one template outside the frames"
    return templates, color_to_template


def _scale_patch(patch: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(patch, np.ones((scale, scale), dtype=int))


def _match_rotation(template: np.ndarray, inner_shape: tuple[int, int], seeds: list[tuple[int, int, int]]) -> np.ndarray:
    inner_h, inner_w = inner_shape
    matches: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(template, k)
        th, tw = rotated.shape
        if inner_h % th != 0 or inner_w % tw != 0:
            continue
        scale_h = inner_h // th
        scale_w = inner_w // tw
        if scale_h != scale_w or scale_h <= 0:
            continue
        enlarged = _scale_patch(rotated, scale_h)
        if all(0 <= r < inner_h and 0 <= c < inner_w and int(enlarged[r, c]) == color for r, c, color in seeds):
            matches.append(enlarged)
    assert len(matches) == 1, "frame seeds must determine a unique rotation"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey frame with a scaled copy of the matching outside template.

    Input structure:
    - The background is black.
    - Several small colored template sprites are scattered above the work area; each template is asymmetric, so its four rotations are distinguishable.
    - Several same-sized grey rectangular frames sit lower in the grid, and each frame interior contains a few colored seed pixels.

    Rule:
    - Detect the grey rectangular frames and the colored template sprites outside them.
    - Use the seed colors inside a frame to identify which outside template it references, then test the four rotations of that template.
    - Enlarge the unique matching rotation by the frame's integer scale factor so each template cell becomes a same-sized square block, and paint that enlarged pattern inside the frame while leaving the rest unchanged.

    Key insight:
    - Seed colors choose the template, but seed positions choose the rotation; both clues are required before the scaled fill can be reconstructed.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames = _extract_frames(g, bg)
    templates, color_to_template = _extract_templates(g, bg, frames)

    out = g.copy()
    for fr in frames:
        inner = g[fr["r0"] + 1 : fr["r1"], fr["c0"] + 1 : fr["c1"]]
        seed_pts = np.argwhere(inner != bg)
        assert len(seed_pts) > 0, "each frame must contain at least one seed pixel"
        seeds = [(int(r), int(c), int(inner[r, c])) for r, c in seed_pts]
        template_ids = {color_to_template[color] for _, _, color in seeds}
        assert len(template_ids) == 1, "all seeds in a frame must point to the same template"
        template = templates[next(iter(template_ids))]
        enlarged = _match_rotation(template, inner.shape, seeds)
        out[fr["r0"] + 1 : fr["r1"], fr["c0"] + 1 : fr["c1"]] = enlarged
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _mask_has_full_bbox(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    r0, c0, r1, c1 = _bbox(pts)
    return r0 == 0 and c0 == 0 and r1 == mask.shape[0] - 1 and c1 == mask.shape[1] - 1


def _rotation_keys(mask: np.ndarray) -> list[bytes]:
    return [np.rot90(mask, k).tobytes() for k in range(4)]


def _canonical_shape(mask: np.ndarray) -> bytes:
    return min(_rotation_keys(mask))


def _random_connected_mask(side: int, rng: random.Random) -> np.ndarray | None:
    cells = [(r, c) for r in range(side) for c in range(side)]
    for _ in range(250):
        target = rng.randint(side + 2, side * side - 1)
        start = rng.choice(cells)
        chosen = {start}
        frontier = {start}
        while len(chosen) < target and frontier:
            base = rng.choice(tuple(frontier))
            nbrs = []
            for dr, dc in DIR4:
                nr, nc = base[0] + dr, base[1] + dc
                if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in chosen:
                    nbrs.append((nr, nc))
            if not nbrs:
                frontier.remove(base)
                continue
            nxt = rng.choice(nbrs)
            chosen.add(nxt)
            frontier.add(nxt)
        mask = np.zeros((side, side), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        if not _mask_connected(mask):
            continue
        if not _mask_has_full_bbox(mask):
            continue
        if len(set(_rotation_keys(mask))) != 4:
            continue
        border = np.zeros_like(mask)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.array_equal(mask, border):
            continue
        return mask
    return None


def _colorize_mask(mask: np.ndarray, colors: tuple[int, int], rng: random.Random) -> np.ndarray:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    rng.shuffle(pts)
    accent_count = rng.randint(1, max(1, len(pts) - 1))
    accent = set(pts[:accent_count])
    patch = np.zeros(mask.shape, dtype=int)
    for r, c in pts:
        patch[r, c] = colors[1] if (r, c) in accent else colors[0]
    return patch


def _partitions(total: int) -> list[tuple[int, ...]]:
    out = [(total,)]
    if total >= 2:
        for left in range(1, total):
            out.append((left, total - left))
    return out


def _distribute(extra: int, slots: int, rng: random.Random) -> list[int]:
    parts = [0] * slots
    for _ in range(extra):
        parts[rng.randrange(slots)] += 1
    return parts


def _row_positions(width: int, count: int, obj_w: int, min_gap: int, rng: random.Random) -> list[int]:
    if count == 0:
        return []
    usable = width - 2
    base = count * obj_w + (count - 1) * min_gap
    extra = usable - base
    assert extra >= 0
    pads = _distribute(extra, count + 1, rng)
    xs: list[int] = []
    x = 1 + pads[0]
    for i in range(count):
        xs.append(x)
        x += obj_w
        if i < count - 1:
            x += min_gap + pads[i + 1]
    return xs


def _template_from_mask(mask: np.ndarray, colors: tuple[int, int], rng: random.Random) -> np.ndarray:
    for _ in range(80):
        patch = _colorize_mask(mask, colors, rng)
        if len(set(_rotation_keys(patch != BG))) == 4:
            return patch
    raise RuntimeError("failed to colorize template")


def _seed_subset_for_rotation(template: np.ndarray, rotation: int, scale: int, rng: random.Random) -> list[tuple[int, int, int]] | None:
    rotated = np.rot90(template, rotation)
    active = [(int(r), int(c), int(rotated[r, c])) for r, c in np.argwhere(rotated != BG)]
    off_r = rng.randrange(scale)
    off_c = rng.randrange(scale)
    candidates = [(r * scale + off_r, c * scale + off_c, color) for r, c, color in active]
    rng.shuffle(candidates)
    max_size = min(5, len(candidates))
    for size in range(2, max_size + 1):
        combos = list(itertools.combinations(candidates, size))
        rng.shuffle(combos)
        for subset in combos[:80]:
            matches = []
            for k in range(4):
                enlarged = _scale_patch(np.rot90(template, k), scale)
                if all(int(enlarged[r, c]) == color for r, c, color in subset):
                    matches.append(k)
            if matches == [rotation]:
                return list(subset)
    return None


def _layout_positions(
    obj_h: int,
    obj_w: int,
    row_lengths: tuple[int, ...],
    width: int,
    y_starts: list[int],
    gap: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for y, count in zip(y_starts, row_lengths):
        xs = _row_positions(width, count, obj_w, gap, rng)
        for x in xs:
            out.append((y, x))
    return out


def _build_layout(side: int, scale: int, num_templates: int, num_frames: int, rng: random.Random):
    ext = side * scale + 2
    t_gap_h = 2
    f_gap_h = 2
    feasible = []
    for t_rows in _partitions(num_templates):
        for f_rows in _partitions(num_frames):
            min_w = max(
                max(count * side + (count - 1) * t_gap_h for count in t_rows),
                max(count * ext + (count - 1) * f_gap_h for count in f_rows),
            ) + 2
            min_h = 1 + len(t_rows) * side + (len(t_rows) - 1) * 1 + 3 + len(f_rows) * ext + (len(f_rows) - 1) * 2 + 1
            if min_w <= 30 and min_h <= 30:
                feasible.append((t_rows, f_rows, min_w, min_h))
    if not feasible:
        return None
    t_rows, f_rows, min_w, min_h = rng.choice(feasible)
    width = rng.randint(min_w, 30)
    height = rng.randint(min_h, 30)

    base_gaps = [1] + [1] * (len(t_rows) - 1) + [3] + [2] * (len(f_rows) - 1) + [1]
    extra = height - (sum(base_gaps) + len(t_rows) * side + len(f_rows) * ext)
    pads = [base + add for base, add in zip(base_gaps, _distribute(extra, len(base_gaps), rng))]

    t_y: list[int] = []
    y = pads[0]
    for i in range(len(t_rows)):
        t_y.append(y)
        y += side
        if i < len(t_rows) - 1:
            y += pads[1 + i]
    middle_idx = len(t_rows)
    y += pads[middle_idx]

    f_y: list[int] = []
    for i in range(len(f_rows)):
        f_y.append(y)
        y += ext
        if i < len(f_rows) - 1:
            y += pads[middle_idx + 1 + i]

    template_pos = _layout_positions(side, side, t_rows, width, t_y, t_gap_h, rng)
    frame_pos = _layout_positions(ext, ext, f_rows, width, f_y, f_gap_h, rng)
    return height, width, template_pos, frame_pos


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    side = rng.choice(TEMPLATE_SIDE_CHOICES)
    scale = rng.choice(SCALE_CHOICES)
    num_templates = rng.randint(2, 4)
    num_frames = rng.randint(2, 4)
    layout = _build_layout(side, scale, num_templates, num_frames, rng)
    if layout is None:
        return None
    height, width, template_pos, frame_pos = layout
    assert len(template_pos) == num_templates and len(frame_pos) == num_frames

    pool = TEMPLATE_COLOR_POOL[:]
    rng.shuffle(pool)
    color_pairs = [tuple(pool[2 * i : 2 * i + 2]) for i in range(num_templates)]

    templates: list[np.ndarray] = []
    used_shapes: set[bytes] = set()
    for i in range(num_templates):
        mask = _random_connected_mask(side, rng)
        if mask is None:
            return None
        signature = _canonical_shape(mask)
        if signature in used_shapes:
            return None
        used_shapes.add(signature)
        templates.append(_template_from_mask(mask, color_pairs[i], rng))

    grid = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)

    for patch, (r0, c0) in zip(templates, template_pos):
        h, w = patch.shape
        grid[r0 : r0 + h, c0 : c0 + w] = patch
        expected[r0 : r0 + h, c0 : c0 + w] = patch

    for idx, (r0, c0) in enumerate(frame_pos):
        grid[r0 : r0 + side * scale + 2, c0 : c0 + side * scale + 2] = BG
        expected[r0 : r0 + side * scale + 2, c0 : c0 + side * scale + 2] = BG
        grid[r0, c0 : c0 + side * scale + 2] = FRAME_COLOR
        grid[r0 + side * scale + 1, c0 : c0 + side * scale + 2] = FRAME_COLOR
        grid[r0 : r0 + side * scale + 2, c0] = FRAME_COLOR
        grid[r0 : r0 + side * scale + 2, c0 + side * scale + 1] = FRAME_COLOR
        expected[r0, c0 : c0 + side * scale + 2] = FRAME_COLOR
        expected[r0 + side * scale + 1, c0 : c0 + side * scale + 2] = FRAME_COLOR
        expected[r0 : r0 + side * scale + 2, c0] = FRAME_COLOR
        expected[r0 : r0 + side * scale + 2, c0 + side * scale + 1] = FRAME_COLOR

        template_idx = rng.randrange(num_templates)
        rotation = rng.randrange(4)
        seeds = _seed_subset_for_rotation(templates[template_idx], rotation, scale, rng)
        if seeds is None:
            return None
        enlarged = _scale_patch(np.rot90(templates[template_idx], rotation), scale)
        expected[r0 + 1 : r0 + 1 + side * scale, c0 + 1 : c0 + 1 + side * scale] = enlarged
        for sr, sc, color in seeds:
            grid[r0 + 1 + sr, c0 + 1 + sc] = color

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frame-template orientation puzzle")
