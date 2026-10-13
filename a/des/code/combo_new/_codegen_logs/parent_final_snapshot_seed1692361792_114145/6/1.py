import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)
SIDES = ("top", "right", "bottom", "left")
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIZE_CHOICES = ((3, 3), (3, 4), (4, 3), (4, 4))

MAX_SIDE = 30
TEMPLATE_COUNT_CHOICES = (2, 3)
FRAME_COUNT_CHOICES = (2, 3)
TEMPLATE_GAP = 2
FRAME_GAP = 2
TOP_TO_FRAMES_GAP = 3
ROW_GAP = 2
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _unique_side(r: int, c: int, h: int, w: int) -> str:
    sides: list[str] = []
    if r == 0:
        sides.append("top")
    if r == h - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == w - 1:
        sides.append("right")
    assert len(sides) == 1, "marker or seed side must be unique"
    return sides[0]


def _rotate_side_ccw(side: str, k: int) -> str:
    idx = SIDES.index(side)
    return SIDES[(idx - k) % 4]


def _rotation_to_match(source_side: str, target_side: str) -> int:
    for k in range(4):
        if _rotate_side_ccw(source_side, k) == target_side:
            return k
    raise AssertionError("no rotation aligns the template marker to the seed side")


def _extract_templates(
    grid: np.ndarray, frames: list[tuple[int, int, int, int]]
) -> dict[int, tuple[np.ndarray, str]]:
    covered = np.zeros(grid.shape, dtype=bool)
    for r0, c0, r1, c1 in frames:
        covered[r0 : r1 + 1, c0 : c1 + 1] = True

    comps = _components((grid != BLACK) & (grid != MAROON) & ~covered)
    templates: dict[int, tuple[np.ndarray, str]] = {}
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        colors = {int(v) for v in np.unique(patch) if int(v) != BLACK}
        assert GREY in colors, "each template must contain a grey marker"
        main_colors = colors - {GREY}
        assert len(main_colors) == 1, "each template must have exactly one main color"
        color = next(iter(main_colors))
        assert color not in templates, "template colors must be unique"

        grey_pts = np.argwhere(patch == GREY)
        assert len(grey_pts) == 1, "each template must have exactly one grey marker"
        gr, gc = map(int, grey_pts[0])
        side = _unique_side(gr, gc, patch.shape[0], patch.shape[1])
        templates[color] = (patch, side)

    assert templates, "expected at least one top template"
    return templates


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for coords in _components(grid == MAROON):
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        expected = set()
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if rr in (r0, r1) or cc in (c0, c1):
                    expected.add((rr, cc))
        if set(coords) == expected:
            frames.append((r0, c0, r1, c1))
    assert frames, "expected at least one maroon rectangular frame"
    frames.sort()
    return frames


def _seed_info(interior: np.ndarray) -> tuple[int, str]:
    coords = np.argwhere(interior != BLACK)
    assert len(coords) > 0, "frame interior must contain a seed"
    colors = {int(interior[r, c]) for r, c in coords}
    assert len(colors) == 1, "seed must use exactly one color"
    color = next(iter(colors))

    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    h, w = interior.shape
    distances = {
        "top": r0,
        "bottom": (h - 1) - r1,
        "left": c0,
        "right": (w - 1) - c1,
    }
    best = min(distances.values())
    sides = [side for side, dist in distances.items() if dist == best]
    assert len(sides) == 1, "seed direction must be unambiguous"
    return color, sides[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed region with the matching top template after rotating it to the seed direction.

    Input structure:
    - The background is black.
    - The upper part contains several disconnected small templates made from one main color plus a single grey marker pixel on one side of the template boundary.
    - The lower part contains several maroon rectangular frames; each frame encloses black space plus a small seed blob whose color matches exactly one template.

    Rule:
    - For each maroon frame, read the enclosed seed color and locate the unique template with the same main color.
    - Determine which side of the frame interior the seed blob is closest to, and which side of the template bbox holds the grey marker.
    - Rotate the template so that the grey marker side matches the seed side, then scale the rotated template by one integer factor so it exactly fits the frame interior.
    - Replace the frame interior with that scaled pattern, keeping the maroon border and every pixel outside the frames unchanged.

    Key insight:
    - The seed color selects the template, while the seed position independently encodes the rotation needed before scaling the template into the frame.
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    templates = _extract_templates(g, frames)

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_color, seed_side = _seed_info(inner)
        patch, marker_side = templates[seed_color]
        rot_k = _rotation_to_match(marker_side, seed_side)
        rotated = np.rot90(patch, rot_k)

        ih, iw = inner.shape
        th, tw = rotated.shape
        assert ih % th == 0 and iw % tw == 0, "frame size must be an integer multiple of its template"
        scale_h = ih // th
        scale_w = iw // tw
        assert scale_h == scale_w and scale_h >= 1, "template scale must be a positive integer"

        scaled = np.repeat(np.repeat(rotated, scale_h, axis=0), scale_w, axis=1)
        out[r0 + 1 : r1, c0 + 1 : c1] = scaled

    return out


def _choose_marker_cell(rng: random.Random, h: int, w: int, side: str) -> tuple[int, int]:
    if side == "top":
        return 0, rng.randint(1, w - 2)
    if side == "bottom":
        return h - 1, rng.randint(1, w - 2)
    if side == "left":
        return rng.randint(1, h - 2), 0
    if side == "right":
        return rng.randint(1, h - 2), w - 1
    raise ValueError(side)


def _make_template_patch(rng: random.Random, color: int, side: str) -> np.ndarray:
    for _ in range(200):
        h, w = rng.choice(SIZE_CHOICES)
        patch = np.full((h, w), color, dtype=int)
        mr, mc = _choose_marker_cell(rng, h, w, side)
        patch[mr, mc] = GREY

        candidates = [(r, c) for r in range(h) for c in range(w) if (r, c) != (mr, mc)]
        rng.shuffle(candidates)
        target_removals = rng.randint(1, min(4, len(candidates) - 3))
        removed = 0
        for r, c in candidates:
            if removed >= target_removals:
                break
            old = int(patch[r, c])
            patch[r, c] = BLACK
            nonblack = patch != BLACK
            if (
                not _is_connected(nonblack)
                or np.any(np.all(~nonblack, axis=0))
                or np.any(np.all(~nonblack, axis=1))
                or np.count_nonzero(nonblack) < 5
            ):
                patch[r, c] = old
                continue
            removed += 1

        if removed == 0:
            continue

        grey_pts = np.argwhere(patch == GREY)
        if len(grey_pts) != 1:
            continue
        gr, gc = map(int, grey_pts[0])
        if _unique_side(gr, gc, h, w) != side:
            continue
        if {int(v) for v in np.unique(patch) if int(v) != BLACK} != {color, GREY}:
            continue
        return patch
    raise RuntimeError("failed to build a valid template patch")


def _render_seed_patch(rng: random.Random, h: int, w: int, side: str, color: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    use_pair = rng.random() < 0.5
    if side in ("top", "bottom") and w < 4:
        use_pair = False
    if side in ("left", "right") and h < 4:
        use_pair = False

    if side == "top":
        row = 0
        if use_pair:
            start = (w - 2) // 2
            patch[row, start : start + 2] = color
        else:
            patch[row, w // 2] = color
    elif side == "bottom":
        row = h - 1
        if use_pair:
            start = (w - 2) // 2
            patch[row, start : start + 2] = color
        else:
            patch[row, w // 2] = color
    elif side == "left":
        col = 0
        if use_pair:
            start = (h - 2) // 2
            patch[start : start + 2, col] = color
        else:
            patch[h // 2, col] = color
    else:
        col = w - 1
        if use_pair:
            start = (h - 2) // 2
            patch[start : start + 2, col] = color
        else:
            patch[h // 2, col] = color
    return patch


def _frame_patch(rng: random.Random, inner_h: int, inner_w: int, side: str, color: int) -> np.ndarray:
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = MAROON
    patch[-1, :] = MAROON
    patch[:, 0] = MAROON
    patch[:, -1] = MAROON
    patch[1:-1, 1:-1] = _render_seed_patch(rng, inner_h, inner_w, side, color)
    return patch


def _row_partitions(n: int) -> list[tuple[int, ...]]:
    if n == 1:
        return [(1,)]
    if n == 2:
        return [(2,), (1, 1)]
    if n == 3:
        return [(3,), (2, 1), (1, 2), (1, 1, 1)]
    raise ValueError("unsupported frame count")


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_templates = rng.choice(TEMPLATE_COUNT_CHOICES)
    n_frames = rng.choice([n for n in FRAME_COUNT_CHOICES if n >= n_templates])

    colors = rng.sample(COLOR_POOL, n_templates)
    marker_sides = rng.sample(SIDES, n_templates)

    templates = []
    for color, marker_side in zip(colors, marker_sides):
        patch = _make_template_patch(rng, color, marker_side)
        templates.append({"color": color, "marker_side": marker_side, "patch": patch})

    frame_template_indices = list(range(n_templates))
    while len(frame_template_indices) < n_frames:
        frame_template_indices.append(rng.randrange(n_templates))
    rng.shuffle(frame_template_indices)

    seed_sides = rng.sample(SIDES, n_frames)
    frame_specs = []
    for idx, seed_side in zip(frame_template_indices, seed_sides):
        template = templates[idx]
        rot_k = _rotation_to_match(template["marker_side"], seed_side)
        rotated = np.rot90(template["patch"], rot_k)

        scale_choices = [1, 2]
        if n_frames == 2 and max(rotated.shape) == 3 and rng.random() < 0.25:
            scale_choices.append(3)
        scale = rng.choice(scale_choices)

        inner_h = int(rotated.shape[0]) * scale
        inner_w = int(rotated.shape[1]) * scale
        patch = _frame_patch(rng, inner_h, inner_w, seed_side, int(template["color"]))
        frame_specs.append({"patch": patch, "h": patch.shape[0], "w": patch.shape[1]})

    template_width = sum(item["patch"].shape[1] for item in templates) + TEMPLATE_GAP * (n_templates - 1)
    template_height = max(item["patch"].shape[0] for item in templates)

    fitting_layouts = []
    for partition in _row_partitions(n_frames):
        rows = []
        start = 0
        for count in partition:
            row_items = frame_specs[start : start + count]
            start += count
            row_width = sum(item["w"] for item in row_items) + FRAME_GAP * (len(row_items) - 1)
            row_height = max(item["h"] for item in row_items)
            rows.append({"items": row_items, "width": row_width, "height": row_height})

        content_width = max(template_width, max(row["width"] for row in rows))
        content_height = template_height + TOP_TO_FRAMES_GAP + sum(row["height"] for row in rows)
        content_height += ROW_GAP * (len(rows) - 1)
        if content_width + 2 <= MAX_SIDE and content_height + 2 <= MAX_SIDE:
            fitting_layouts.append((rows, content_width, content_height))

    if not fitting_layouts:
        return None

    rows, content_width, content_height = rng.choice(fitting_layouts)
    extra_w = rng.randint(0, min(4, MAX_SIDE - (content_width + 2)))
    extra_h = rng.randint(0, min(4, MAX_SIDE - (content_height + 2)))
    grid_w = content_width + 2 + extra_w
    grid_h = content_height + 2 + extra_h
    grid = np.zeros((grid_h, grid_w), dtype=int)

    content_left = 1 + rng.randint(0, extra_w)
    content_top = 1 + rng.randint(0, extra_h)

    template_left = content_left + rng.randint(0, content_width - template_width)
    x = template_left
    for item in templates:
        patch = item["patch"]
        y = content_top + rng.randint(0, template_height - patch.shape[0])
        _paste(grid, patch, y, x)
        x += patch.shape[1] + TEMPLATE_GAP

    y = content_top + template_height + TOP_TO_FRAMES_GAP
    for row in rows:
        row_left = content_left + rng.randint(0, content_width - row["width"])
        x = row_left
        for item in row["items"]:
            dy = rng.randint(0, row["height"] - item["h"])
            _paste(grid, item["patch"], y + dy, x)
            x += item["w"] + FRAME_GAP
        y += row["height"] + ROW_GAP

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
