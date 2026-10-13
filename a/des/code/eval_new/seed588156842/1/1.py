import random
from collections import Counter, deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GAP_DIRECTIONS = ["right", "left", "up", "down"]
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _transform_patch(patch: np.ndarray, direction: str) -> np.ndarray:
    if direction == "right":
        return patch.copy()
    if direction == "left":
        return np.fliplr(patch)
    if direction == "up":
        return np.rot90(patch, -1)
    if direction == "down":
        return np.rot90(patch, 1)
    raise AssertionError(f"unknown direction: {direction}")


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    mask = grid != bg
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    for comp in _components(mask):
        pts = {tuple(map(int, p)) for p in comp}
        if not any(corner in pts for corner in corners):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if sub.shape[1] != 2:
            continue
        if comp.shape[0] != sub.size or np.any(sub == bg):
            continue
        mapping = {}
        for rr in range(sub.shape[0]):
            mapping[int(sub[rr, 0])] = int(sub[rr, 1])
        legend_mask = np.zeros_like(mask)
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        return mapping, legend_mask
    raise AssertionError("legend not found")


def _gap_run(values: np.ndarray, bg: int, color: int) -> tuple[int, int] | None:
    bg_idxs = [i for i, v in enumerate(values.tolist()) if int(v) == bg]
    if not bg_idxs:
        if np.all(values == color):
            return None
        return None
    if any(int(v) not in (bg, color) for v in values.tolist()):
        return None
    if bg_idxs != list(range(bg_idxs[0], bg_idxs[-1] + 1)):
        return None
    if not (1 <= len(bg_idxs) <= 3):
        return None
    for i, v in enumerate(values.tolist()):
        if i not in bg_idxs and int(v) != color:
            return None
    return int(bg_idxs[0]), int(bg_idxs[-1])


def _frame_info(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict | None:
    colors = {int(grid[r, c]) for r, c in coords}
    if len(colors) != 1:
        return None
    color = next(iter(colors))
    r0, c0, r1, c1 = _bbox(coords)
    hh, ww = r1 - r0 + 1, c1 - c0 + 1
    if hh < 4 or ww < 4:
        return None

    sub = np.full((hh, ww), bg, dtype=int)
    for r, c in coords:
        sub[int(r) - r0, int(c) - c0] = color

    if np.any(sub[1:-1, 1:-1] != bg):
        return None
    if any(
        int(sub[rr, cc]) != color
        for rr, cc in [(0, 0), (0, ww - 1), (hh - 1, 0), (hh - 1, ww - 1)]
    ):
        return None

    candidates = []
    checks = [
        ("up", sub[0, 1:-1]),
        ("down", sub[hh - 1, 1:-1]),
        ("left", sub[1:-1, 0]),
        ("right", sub[1:-1, ww - 1]),
    ]
    for direction, segment in checks:
        run = _gap_run(segment, bg, color)
        if run is not None:
            candidates.append((direction, run))
        elif not np.all(segment == color):
            return None

    if len(candidates) != 1:
        return None

    direction, (s0, s1) = candidates[0]
    gap_coords = []
    if direction == "up":
        gap_coords = [(r0, c0 + 1 + cc) for cc in range(s0, s1 + 1)]
    elif direction == "down":
        gap_coords = [(r1, c0 + 1 + cc) for cc in range(s0, s1 + 1)]
    elif direction == "left":
        gap_coords = [(r0 + 1 + rr, c0) for rr in range(s0, s1 + 1)]
    else:
        gap_coords = [(r0 + 1 + rr, c1) for rr in range(s0, s1 + 1)]

    expected_area = 2 * hh + 2 * ww - 4 - len(gap_coords)
    if int(coords.shape[0]) != expected_area:
        return None

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "direction": direction,
        "gap_coords": gap_coords,
    }


def _sprite_info(grid: np.ndarray, coords: np.ndarray, bg: int, legend_keys: set[int]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    hh, ww = r1 - r0 + 1, c1 - c0 + 1
    if hh > 4 or ww > 4:
        return None
    colors = {int(grid[r, c]) for r, c in coords}
    if len(colors) != 2:
        return None
    outline_candidates = sorted(color for color in colors if color in legend_keys)
    if len(outline_candidates) != 1:
        return None
    outline_color = outline_candidates[0]
    fill_color = next(color for color in colors if color != outline_color)
    patch = np.full((hh, ww), bg, dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(grid[r, c])
    return {
        "outline_color": outline_color,
        "fill_color": fill_color,
        "coords": [(int(r), int(c)) for r, c in coords],
        "patch": patch,
    }


def _anchor_coords(
    patch: np.ndarray, outline_color: int, bg: int, direction: str
) -> list[tuple[int, int]]:
    outline_pts = np.argwhere(patch == outline_color)
    if outline_pts.size == 0:
        raise AssertionError("sprite has no outline pixels")
    if direction == "right":
        anchor_value = int(outline_pts[:, 1].max())
        coords = [(int(r), int(c)) for r, c in outline_pts if int(c) == anchor_value]
        coords.sort()
        return coords
    if direction == "left":
        anchor_value = int(outline_pts[:, 1].min())
        coords = [(int(r), int(c)) for r, c in outline_pts if int(c) == anchor_value]
        coords.sort()
        return coords
    if direction == "up":
        anchor_value = int(outline_pts[:, 0].min())
        coords = [(int(r), int(c)) for r, c in outline_pts if int(r) == anchor_value]
        coords.sort(key=lambda p: (p[1], p[0]))
        return coords
    if direction == "down":
        anchor_value = int(outline_pts[:, 0].max())
        coords = [(int(r), int(c)) for r, c in outline_pts if int(r) == anchor_value]
        coords.sort(key=lambda p: (p[1], p[0]))
        return coords
    raise AssertionError(f"unknown direction: {direction}")


def _paste_sprite(
    out: np.ndarray,
    frame: dict,
    sprite_patch: np.ndarray,
    outline_color: int,
    frame_color: int,
    bg: int,
) -> None:
    transformed = _transform_patch(sprite_patch, frame["direction"])
    gap_coords = list(frame["gap_coords"])
    if frame["direction"] in {"right", "left"}:
        gap_coords.sort()
    else:
        gap_coords.sort(key=lambda p: (p[1], p[0]))
    anchors = _anchor_coords(transformed, outline_color, bg, frame["direction"])
    if len(anchors) != len(gap_coords):
        raise AssertionError("gap length does not match transformed sprite anchor")

    dr = gap_coords[0][0] - anchors[0][0]
    dc = gap_coords[0][1] - anchors[0][1]
    for (ar, ac), (gr, gc) in zip(anchors, gap_coords):
        if ar + dr != gr or ac + dc != gc:
            raise AssertionError("sprite anchor does not align with frame gap")

    h, w = out.shape
    for rr, cc in np.argwhere(transformed != bg):
        tr = int(rr + dr)
        tc = int(cc + dc)
        if not (0 <= tr < h and 0 <= tc < w):
            raise AssertionError("pasted sprite is out of bounds")
        value = int(transformed[rr, cc])
        out[tr, tc] = frame_color if value == outline_color else value


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each two-color fragment into the frame with the matching border color and orient it to plug the frame gap.

    Input structure:
    - The background is the most common color.
    - One corner contains a small color legend whose rows are 2 cells wide: left cell = fragment outline color, right cell = frame border color.
    - Elsewhere are several monochrome rectangular frames with a single contiguous gap on one side, plus several small 2-color fragment sprites.

    Rule:
    - Read the corner legend to build the mapping from fragment outline color to frame border color.
    - Detect each frame's missing side by finding the only contiguous background run on its border.
    - Match each fragment to the frame whose border color is paired with that fragment's outline color.
    - Transform the fragment by gap direction: right = unchanged, left = horizontal flip, up = 90 degrees clockwise, down = 90 degrees counterclockwise.
    - Place the transformed fragment so its outline pixels sit on the frame gap, recolor those outline pixels to the frame border color, keep fill pixels unchanged, and erase the fragment's original location.

    Key insight:
    The legend links outline colors to frame colors, while the side containing the border gap determines the exact flip or rotation needed before docking.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legend_map, legend_mask = _find_legend(g, bg)

    objects = _components((g != bg) & (~legend_mask))
    frames_by_color: dict[int, dict] = {}
    sprites_by_outline: dict[int, dict] = {}

    for comp in objects:
        frame = _frame_info(g, comp, bg)
        if frame is not None:
            color = int(frame["color"])
            if color in frames_by_color:
                raise AssertionError("frame colors must be unique")
            frames_by_color[color] = frame
            continue
        sprite = _sprite_info(g, comp, bg, set(legend_map))
        if sprite is not None:
            outline = int(sprite["outline_color"])
            if outline in sprites_by_outline:
                raise AssertionError("outline colors must be unique")
            sprites_by_outline[outline] = sprite

    out = g.copy()
    for sprite in sprites_by_outline.values():
        for r, c in sprite["coords"]:
            out[r, c] = bg

    for outline_color, frame_color in legend_map.items():
        if outline_color not in sprites_by_outline or frame_color not in frames_by_color:
            raise AssertionError("legend pair is missing its sprite or frame")
        _paste_sprite(
            out,
            frames_by_color[frame_color],
            sprites_by_outline[outline_color]["patch"],
            outline_color,
            frame_color,
            bg,
        )

    return out


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expanded_bbox(
    bbox: tuple[int, int, int, int], pad: int, h: int, w: int
) -> tuple[int, int, int, int]:
    return (
        max(0, bbox[0] - pad),
        max(0, bbox[1] - pad),
        min(h - 1, bbox[2] + pad),
        min(w - 1, bbox[3] + pad),
    )


def _can_place(
    bbox: tuple[int, int, int, int], reserved: list[tuple[int, int, int, int]], pad: int, h: int, w: int
) -> bool:
    expanded = _expanded_bbox(bbox, pad, h, w)
    return all(not _rects_overlap(expanded, other) for other in reserved)


def _draw_frame(grid: np.ndarray, frame: dict, bg: int) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    color = int(frame["color"])
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    for r, c in frame["gap_coords"]:
        grid[r, c] = bg


def _random_sprite_patch(rng: random.Random, height: int, outline_color: int, fill_color: int) -> np.ndarray:
    for _ in range(40):
        width = rng.choice([3, 4])
        patch = np.full((height, width), outline_color, dtype=int)
        middle = [(r, c) for r in range(height) for c in range(1, width - 1)]
        chosen_count = rng.randint(1, len(middle))
        for r, c in rng.sample(middle, chosen_count):
            patch[r, c] = fill_color
        if len({_transform_patch(patch, d).tobytes() for d in GAP_DIRECTIONS}) >= 2:
            return patch
    patch = np.full((height, 3), outline_color, dtype=int)
    patch[-1, 1] = fill_color
    return patch


def _make_frame_spec(
    rng: random.Random, frame_color: int, direction: str, gap_len: int, sprite_patch: np.ndarray
) -> dict:
    transformed = _transform_patch(sprite_patch, direction)
    th, tw = transformed.shape
    if direction in {"right", "left"}:
        min_h = gap_len + 2
        min_w = tw + 1
    else:
        min_h = th + 1
        min_w = gap_len + 2
    outer_h = rng.randint(min_h, min(min_h + 3, 10))
    outer_w = rng.randint(min_w, min(min_w + 4, 10))

    if direction in {"left", "right"}:
        start = rng.randint(1, outer_h - gap_len - 1)
    else:
        start = rng.randint(1, outer_w - gap_len - 1)

    bbox = (0, 0, outer_h - 1, outer_w - 1)
    frame = {"color": frame_color, "bbox": bbox, "direction": direction, "gap_coords": []}
    if direction == "right":
        frame["gap_coords"] = [(1 + start_offset, outer_w - 1) for start_offset in range(gap_len)]
        frame["gap_coords"] = [(start + k, outer_w - 1) for k in range(gap_len)]
    elif direction == "left":
        frame["gap_coords"] = [(start + k, 0) for k in range(gap_len)]
    elif direction == "up":
        frame["gap_coords"] = [(0, start + k) for k in range(gap_len)]
    else:
        frame["gap_coords"] = [(outer_h - 1, start + k) for k in range(gap_len)]
    return frame


def _shift_frame(frame: dict, top: int, left: int) -> dict:
    r0, c0, r1, c1 = frame["bbox"]
    return {
        "color": int(frame["color"]),
        "bbox": (r0 + top, c0 + left, r1 + top, c1 + left),
        "direction": frame["direction"],
        "gap_coords": [(r + top, c + left) for r, c in frame["gap_coords"]],
    }


def _expected_output(
    inp: np.ndarray,
    bg: int,
    legend_rows: list[tuple[int, int]],
    frame_specs: list[dict],
    sprite_specs: list[dict],
) -> np.ndarray:
    out = inp.copy()
    for sprite in sprite_specs:
        patch = sprite["patch"]
        top, left = sprite["top"], sprite["left"]
        h, w = patch.shape
        out[top : top + h, left : left + w] = bg
    frame_by_color = {int(frame["color"]): frame for frame in frame_specs}
    sprite_by_outline = {int(sprite["outline_color"]): sprite for sprite in sprite_specs}
    for outline_color, frame_color in legend_rows:
        sprite = sprite_by_outline[int(outline_color)]
        frame = frame_by_color[int(frame_color)]
        _paste_sprite(out, frame, sprite["patch"], int(outline_color), int(frame_color), bg)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_pairs = rng.randint(2, 4)
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    bg = rng.randrange(10)

    usable_colors = [c for c in range(10) if c != bg]
    picked = rng.sample(usable_colors, 2 * n_pairs)
    outline_colors = picked[:n_pairs]
    frame_colors = picked[n_pairs:]
    fill_palette = [c for c in usable_colors if c not in outline_colors]

    grid = np.full((h, w), bg, dtype=int)

    legend_corner = rng.choice(["tl", "tr", "bl", "br"])
    legend_bbox = {
        "tl": (0, 0, n_pairs - 1, 1),
        "tr": (0, w - 2, n_pairs - 1, w - 1),
        "bl": (h - n_pairs, 0, h - 1, 1),
        "br": (h - n_pairs, w - 2, h - 1, w - 1),
    }[legend_corner]
    reserved = [_expanded_bbox(legend_bbox, 1, h, w)]

    legend_rows = list(zip(outline_colors, frame_colors))
    rng.shuffle(legend_rows)
    lr0, lc0, _, _ = legend_bbox
    for idx, (outline_color, frame_color) in enumerate(legend_rows):
        grid[lr0 + idx, lc0] = outline_color
        grid[lr0 + idx, lc0 + 1] = frame_color

    latent_pairs = []
    for outline_color, frame_color in zip(outline_colors, frame_colors):
        gap_len = rng.choice([2, 3])
        fill_choices = [c for c in fill_palette if c != frame_color]
        fill_color = rng.choice(fill_choices if fill_choices else fill_palette)
        sprite_patch = _random_sprite_patch(rng, gap_len, outline_color, fill_color)
        direction = rng.choice(GAP_DIRECTIONS)
        frame_template = _make_frame_spec(rng, frame_color, direction, gap_len, sprite_patch)
        latent_pairs.append(
            {
                "outline_color": outline_color,
                "frame_color": frame_color,
                "sprite_patch": sprite_patch,
                "direction": direction,
                "frame_template": frame_template,
            }
        )

    rng.shuffle(latent_pairs)

    frame_specs = []
    sprite_specs = []
    for pair in latent_pairs:
        template = pair["frame_template"]
        fr0, fc0, fr1, fc1 = template["bbox"]
        fh, fw = fr1 - fr0 + 1, fc1 - fc0 + 1
        placed = False
        for _ in range(120):
            top = rng.randint(0, h - fh)
            left = rng.randint(0, w - fw)
            bbox = (top, left, top + fh - 1, left + fw - 1)
            if not _can_place(bbox, reserved, 1, h, w):
                continue
            frame = _shift_frame(template, top, left)
            frame_specs.append(frame)
            reserved.append(_expanded_bbox(bbox, 1, h, w))
            placed = True
            break
        if not placed:
            return None

    for pair in latent_pairs:
        patch = pair["sprite_patch"]
        sh, sw = patch.shape
        placed = False
        for _ in range(120):
            top = rng.randint(0, h - sh)
            left = rng.randint(0, w - sw)
            bbox = (top, left, top + sh - 1, left + sw - 1)
            if not _can_place(bbox, reserved, 1, h, w):
                continue
            sprite_specs.append(
                {
                    "outline_color": int(pair["outline_color"]),
                    "frame_color": int(pair["frame_color"]),
                    "patch": patch.copy(),
                    "top": top,
                    "left": left,
                }
            )
            reserved.append(_expanded_bbox(bbox, 1, h, w))
            placed = True
            break
        if not placed:
            return None

    for frame in frame_specs:
        _draw_frame(grid, frame, bg)
    for sprite in sprite_specs:
        patch = sprite["patch"]
        sh, sw = patch.shape
        top, left = int(sprite["top"]), int(sprite["left"])
        grid[top : top + sh, left : left + sw] = patch

    expected = _expected_output(grid, bg, legend_rows, frame_specs, sprite_specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid grid")
