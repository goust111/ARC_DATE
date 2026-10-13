import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
DIRECTIONS = ("up", "down", "left", "right")
PATTERN_LIBRARY = (
    np.array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
)
FRAME_COUNT_RANGE = (2, 4)
FRAME_SIDE_RANGE = (14, 16)
GRID_PADDING = 1
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
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
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _color_components(grid: np.ndarray, active_mask: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not active_mask[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or not active_mask[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            coords = np.array(pts, dtype=int)
            comps.append({"color": color, "coords": coords})
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, r0, c0


def _opening_direction(mask: np.ndarray) -> str:
    scores = {
        "up": int(mask[0].sum()),
        "down": int(mask[-1].sum()),
        "left": int(mask[:, 0].sum()),
        "right": int(mask[:, -1].sum()),
    }
    best = min(scores.values())
    dirs = [name for name, value in scores.items() if value == best]
    assert len(dirs) == 1, "shape orientation must have a unique opening direction"
    return dirs[0]


def _oriented_variant(base_mask: np.ndarray, direction: str) -> np.ndarray:
    for k in range(4):
        candidate = np.rot90(base_mask, k)
        if _opening_direction(candidate) == direction:
            return candidate
    raise AssertionError("rotation not found")


def _detect_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    candidates = []
    specs = (
        ("tl", 0, 1, 1),
        ("tr", 0, 1, -1),
        ("bl", h - 2, h - 1, 1),
        ("br", h - 2, h - 1, -1),
    )
    for _name, r0, r1, step in specs:
        if not (0 <= r0 < h and 0 <= r1 < h):
            continue
        start_c = 0 if step == 1 else w - 1
        cols = []
        c = start_c
        while 0 <= c < w and int(grid[r0, c]) != bg and int(grid[r1, c]) != bg:
            cols.append(c)
            c += step
        if len(cols) < 2:
            continue
        if 0 <= c < w and (int(grid[r0, c]) != bg or int(grid[r1, c]) != bg):
            continue
        probe_r = r1 + 1 if r0 == 0 else r0 - 1
        if 0 <= probe_r < h and any(int(grid[probe_r, cc]) != bg for cc in cols):
            continue
        mapping: dict[int, int] = {}
        ok = True
        for cc in cols:
            key = int(grid[r0, cc])
            val = int(grid[r1, cc])
            if key in mapping and mapping[key] != val:
                ok = False
                break
            mapping[key] = val
        if not ok or len(mapping) != len(cols):
            continue
        mask = np.zeros_like(grid, dtype=bool)
        mask[r0, cols] = True
        mask[r1, cols] = True
        candidates.append((len(cols), mapping, mask))
    assert candidates, "legend not found"
    _, mapping, mask = max(candidates, key=lambda item: item[0])
    return mapping, mask


def _frame_from_component(grid: np.ndarray, coords: np.ndarray, color: int, bg: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    side = r1 - r0 + 1
    if side != c1 - c0 + 1 or side < 6:
        return None
    if int(coords.shape[0]) != 4 * side - 6:
        return None
    coordset = {tuple(map(int, pt)) for pt in coords}
    if any(r not in (r0, r1) and c not in (c0, c1) for r, c in coordset):
        return None

    side_lines = {
        "up": [(r0, c) for c in range(c0 + 1, c1)],
        "down": [(r1, c) for c in range(c0 + 1, c1)],
        "left": [(r, c0) for r in range(r0 + 1, r1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
    }
    all_border = (
        [(r0, c) for c in range(c0, c1 + 1)]
        + [(r1, c) for c in range(c0, c1 + 1)]
        + [(r, c0) for r in range(r0 + 1, r1)]
        + [(r, c1) for r in range(r0 + 1, r1)]
    )
    gap_side = None
    gap_cells: set[tuple[int, int]] = set()
    gap_start = None
    for name, line in side_lines.items():
        missing = [i for i, pos in enumerate(line) if int(grid[pos]) == bg]
        if any(int(grid[pos]) not in (bg, color) for pos in line):
            return None
        if not missing:
            continue
        if gap_side is not None or len(missing) != 2 or missing[1] != missing[0] + 1:
            return None
        gap_side = name
        gap_cells = {line[missing[0]], line[missing[1]]}
        gap_start = int(line[missing[0]][1 if name in ("up", "down") else 0])
    if gap_side is None or gap_start is None:
        return None
    for pos in all_border:
        value = int(grid[pos])
        if pos in gap_cells:
            if value != bg:
                return None
        elif value != color:
            return None
    return {
        "color": color,
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "gap_side": gap_side,
        "gap_start": gap_start,
    }


def _find_frames(grid: np.ndarray, bg: int, legend_mask: np.ndarray) -> tuple[list[dict], np.ndarray]:
    active = (grid != bg) & (~legend_mask)
    frames = []
    frame_mask = np.zeros_like(grid, dtype=bool)
    for comp in _color_components(grid, active):
        info = _frame_from_component(grid, comp["coords"], int(comp["color"]), bg)
        if info is None:
            continue
        frames.append(info)
        frame_mask[comp["coords"][:, 0], comp["coords"][:, 1]] = True
    assert frames, "no frames found"
    return frames, frame_mask


def _seed_set_from_mask(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {
        (top + r, left + c)
        for r in range(mask.shape[0])
        for c in range(mask.shape[1])
        if bool(mask[r, c])
    }


def _extract_seed_for_frame(grid: np.ndarray, frame: dict) -> dict:
    r0, c0, r1, c1 = frame["bbox"]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    comps = _components_mask(interior == GREY)
    assert len(comps) == 1, "each frame must contain exactly one grey seed"
    local = comps[0]
    global_coords = local + np.array([r0 + 1, c0 + 1], dtype=int)
    mask, sr, sc = _tight_mask(global_coords)
    return {
        "coords": global_coords,
        "mask": mask,
        "top": sr,
        "left": sc,
        "direction": _opening_direction(mask),
    }


def _find_sprites(
    grid: np.ndarray,
    bg: int,
    legend_mask: np.ndarray,
    frame_mask: np.ndarray,
) -> list[dict]:
    active = (grid != bg) & (grid != GREY) & (~legend_mask) & (~frame_mask)
    sprites = []
    for comp in _color_components(grid, active):
        mask, r0, c0 = _tight_mask(comp["coords"])
        sprites.append(
            {
                "color": int(comp["color"]),
                "coords": comp["coords"],
                "mask": mask,
                "top": r0,
                "left": c0,
                "direction": _opening_direction(mask),
            }
        )
    assert sprites, "no external sprites found"
    return sprites


def _can_place_sprite(
    frame_box: tuple[int, int, int, int],
    sprite_mask: np.ndarray,
    top: int,
    left: int,
    seed_cells: set[tuple[int, int]],
) -> tuple[bool, str | None]:
    r0, c0, r1, c1 = frame_box
    for sr in range(sprite_mask.shape[0]):
        for sc in range(sprite_mask.shape[1]):
            if not sprite_mask[sr, sc]:
                continue
            gr = top + sr
            gc = left + sc
            if not (r0 + 1 <= gr <= r1 - 1 and c0 + 1 <= gc <= c1 - 1):
                return False, "wall"
            if (gr, gc) in seed_cells:
                return False, "seed"
    return True, None


def _simulate_slide(
    frame_box: tuple[int, int, int, int],
    sprite_mask: np.ndarray,
    gap_side: str,
    gap_start: int,
    seed_cells: set[tuple[int, int]],
) -> tuple[int, int, str] | None:
    r0, c0, r1, c1 = frame_box
    sh, sw = sprite_mask.shape
    if gap_side == "up":
        top, left = r0 + 1, gap_start
        dr, dc = 1, 0
    elif gap_side == "down":
        top, left = r1 - sh, gap_start
        dr, dc = -1, 0
    elif gap_side == "left":
        top, left = gap_start, c0 + 1
        dr, dc = 0, 1
    else:
        top, left = gap_start, c1 - sw
        dr, dc = 0, -1

    ok, _ = _can_place_sprite(frame_box, sprite_mask, top, left, seed_cells)
    if not ok:
        return None
    while True:
        ok, reason = _can_place_sprite(frame_box, sprite_mask, top + dr, left + dc, seed_cells)
        if not ok:
            return top, left, str(reason)
        top += dr
        left += dc


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    legend_map, legend_mask = _detect_legend(g, bg)
    frames, frame_mask = _find_frames(g, bg, legend_mask)
    seeds = [_extract_seed_for_frame(g, frame) for frame in frames]
    sprites = _find_sprites(g, bg, legend_mask, frame_mask)

    sprite_by_direction: dict[str, dict] = {}
    for sprite in sprites:
        direction = sprite["direction"]
        assert direction not in sprite_by_direction, "sprite orientations must be unique"
        sprite_by_direction[direction] = sprite

    out = np.full_like(g, bg)
    for frame, seed in zip(frames, seeds):
        out[frame["coords"][:, 0], frame["coords"][:, 1]] = int(frame["color"])
        direction = seed["direction"]
        assert direction in sprite_by_direction, "missing sprite for seed direction"
        sprite = sprite_by_direction[direction]
        seed_cells = {tuple(map(int, pt)) for pt in seed["coords"]}
        placed = _simulate_slide(
            frame["bbox"],
            sprite["mask"],
            str(frame["gap_side"]),
            int(frame["gap_start"]),
            seed_cells,
        )
        assert placed is not None, "sprite cannot enter its frame"
        top, left, _reason = placed
        new_color = int(legend_map[int(frame["color"])])
        for sr in range(sprite["mask"].shape[0]):
            for sc in range(sprite["mask"].shape[1]):
                if sprite["mask"][sr, sc]:
                    out[top + sr, left + sc] = new_color
    return out


def _mark_box(blocked: np.ndarray, r: int, c: int, h: int, w: int, pad: int = GRID_PADDING) -> None:
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(blocked.shape[0], r + h + pad)
    cc1 = min(blocked.shape[1], c + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _box_free(blocked: np.ndarray, r: int, c: int, h: int, w: int, pad: int = GRID_PADDING) -> bool:
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(blocked.shape[0], r + h + pad)
    cc1 = min(blocked.shape[1], c + w + pad)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _paste_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c]:
                grid[top + r, left + c] = int(color)


def _draw_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    side: int,
    color: int,
    gap_side: str,
    gap_start: int,
) -> None:
    bottom = top + side - 1
    right = left + side - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color
    if gap_side == "up":
        grid[top, gap_start : gap_start + 2] = BG
    elif gap_side == "down":
        grid[bottom, gap_start : gap_start + 2] = BG
    elif gap_side == "left":
        grid[gap_start : gap_start + 2, left] = BG
    else:
        grid[gap_start : gap_start + 2, right] = BG


def _legend_dims(count: int) -> tuple[int, int]:
    return 2, count


def _draw_legend(
    grid: np.ndarray,
    corner: str,
    frame_colors: list[int],
    out_colors: list[int],
) -> None:
    count = len(frame_colors)
    h, w = grid.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - count
    elif corner == "bl":
        r0, c0 = h - 2, 0
    else:
        r0, c0 = h - 2, w - count
    grid[r0, c0 : c0 + count] = np.array(frame_colors, dtype=int)
    grid[r0 + 1, c0 : c0 + count] = np.array(out_colors, dtype=int)


def _seed_layout(side: int, seed_mask: np.ndarray) -> tuple[int, int, set[tuple[int, int]]]:
    inner = side - 2
    top = 1 + (inner - seed_mask.shape[0]) // 2
    left = 1 + (inner - seed_mask.shape[1]) // 2
    return top, left, _seed_set_from_mask(seed_mask, top, left)


def _gap_specs(
    side: int,
    sprite_mask: np.ndarray,
    seed_cells: set[tuple[int, int]],
) -> list[tuple[str, int, str]]:
    frame_box = (0, 0, side - 1, side - 1)
    specs = []
    for gap_side in DIRECTIONS:
        if gap_side in ("up", "down"):
            upper = side - 1 - sprite_mask.shape[1]
        else:
            upper = side - 1 - sprite_mask.shape[0]
        for gap_start in range(1, upper + 1):
            placed = _simulate_slide(frame_box, sprite_mask, gap_side, gap_start, seed_cells)
            if placed is None:
                continue
            _top, _left, reason = placed
            specs.append((gap_side, gap_start, reason))
    return specs


def _choose_object_colors(rng: random.Random, count: int) -> tuple[list[int], list[int], list[int]]:
    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    picked = rng.sample(palette, 2 * count)
    frame_colors = picked[:count]
    out_colors = picked[count:]
    sprite_pool = [c for c in palette if c not in frame_colors]
    sprite_colors = rng.sample(sprite_pool, count)
    return frame_colors, out_colors, sprite_colors


def _random_oriented_mask(rng: random.Random, direction: str) -> np.ndarray:
    return _oriented_variant(rng.choice(PATTERN_LIBRARY), direction)


def _reserve_legend(blocked: np.ndarray, corner: str, count: int) -> None:
    h, w = blocked.shape
    lh, lw = _legend_dims(count)
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - lw
    elif corner == "bl":
        r0, c0 = h - lh, 0
    else:
        r0, c0 = h - lh, w - lw
    _mark_box(blocked, r0, c0, lh, lw)


def _place_boxes(
    rng: random.Random,
    blocked: np.ndarray,
    sizes: list[tuple[int, int]],
    margin_from_edge: int,
) -> list[tuple[int, int]] | None:
    h, w = blocked.shape
    positions = []
    for bh, bw in sizes:
        placed = None
        for _ in range(250):
            r = rng.randint(margin_from_edge, h - bh - margin_from_edge)
            c = rng.randint(margin_from_edge, w - bw - margin_from_edge)
            if not _box_free(blocked, r, c, bh, bw):
                continue
            placed = (r, c)
            _mark_box(blocked, r, c, bh, bw)
            break
        if placed is None:
            return None
        positions.append(placed)
    return positions


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.randint(*FRAME_COUNT_RANGE)
    frame_side = rng.randint(*FRAME_SIDE_RANGE)
    directions = rng.sample(list(DIRECTIONS), frame_count)
    frame_colors, out_colors, sprite_colors = _choose_object_colors(rng, frame_count)
    legend_corner = rng.choice(("tl", "tr", "bl", "br"))

    objects = []
    for direction, frame_color, out_color, sprite_color in zip(
        directions, frame_colors, out_colors, sprite_colors
    ):
        seed_mask = _random_oriented_mask(rng, direction)
        sprite_mask = _random_oriented_mask(rng, direction)
        _seed_top, _seed_left, seed_cells = _seed_layout(frame_side, seed_mask)
        all_specs = _gap_specs(frame_side, sprite_mask, seed_cells)
        if not all_specs:
            return None
        desired_reason = rng.choice(("seed", "wall"))
        matching = [spec for spec in all_specs if spec[2] == desired_reason]
        gap_side, gap_start, stop_reason = rng.choice(matching or all_specs)
        objects.append(
            {
                "direction": direction,
                "frame_color": frame_color,
                "out_color": out_color,
                "sprite_color": sprite_color,
                "seed_mask": seed_mask,
                "sprite_mask": sprite_mask,
                "gap_side": gap_side,
                "gap_start": gap_start,
                "stop_reason": stop_reason,
            }
        )

    base_side = max(32, 2 * frame_side + 4)
    h = rng.randint(base_side, base_side + 10)
    w = rng.randint(base_side, base_side + 10)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    _reserve_legend(blocked, legend_corner, frame_count)

    frame_positions = _place_boxes(
        rng,
        blocked,
        [(frame_side, frame_side)] * frame_count,
        margin_from_edge=1,
    )
    if frame_positions is None:
        return None

    sprite_sizes = [(obj["sprite_mask"].shape[0], obj["sprite_mask"].shape[1]) for obj in objects]
    sprite_positions = _place_boxes(rng, blocked, sprite_sizes, margin_from_edge=1)
    if sprite_positions is None:
        return None

    _draw_legend(grid, legend_corner, frame_colors, out_colors)
    for obj, (fr, fc), (sr, sc) in zip(objects, frame_positions, sprite_positions):
        gap_start_global = int(fc + obj["gap_start"]) if obj["gap_side"] in ("up", "down") else int(
            fr + obj["gap_start"]
        )
        _draw_frame(grid, fr, fc, frame_side, int(obj["frame_color"]), str(obj["gap_side"]), gap_start_global)
        seed_top, seed_left, _seed_cells = _seed_layout(frame_side, obj["seed_mask"])
        _paste_mask(grid, fr + seed_top, fc + seed_left, obj["seed_mask"], GREY)
        _paste_mask(grid, sr, sc, obj["sprite_mask"], int(obj["sprite_color"]))

    if len({obj["direction"] for obj in objects}) != frame_count:
        return None
    if len({obj["frame_color"] for obj in objects}) != frame_count:
        return None
    if len({obj["stop_reason"] for obj in objects}) < 1:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
