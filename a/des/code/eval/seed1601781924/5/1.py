import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return comps


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _frame_info(comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    mask = _component_mask(comp)
    if mask[1:-1, 1:-1].any():
        return None

    side_lines = {
        "top": mask[0, :],
        "bottom": mask[-1, :],
        "left": mask[:, 0],
        "right": mask[:, -1],
    }

    gap_side = None
    gap_interval = None
    for side, line in side_lines.items():
        missing = np.flatnonzero(~line)
        if missing.size == 0:
            continue
        if gap_side is not None:
            return None
        if not np.array_equal(missing, np.arange(missing[0], missing[-1] + 1)):
            return None
        gap_side = side
        gap_interval = (int(missing[0]), int(missing[-1]))

    if gap_side is None or gap_interval is None:
        return None

    gap_len = gap_interval[1] - gap_interval[0] + 1
    expected_area = 2 * h + 2 * w - 4 - gap_len
    if int(mask.sum()) != expected_area:
        return None

    return {
        "color": int(comp["color"]),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap_side": gap_side,
        "gap_interval": gap_interval,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project mirrored sprite copies outside a gapped rectangular frame.

    Each monochrome rectangular frame has exactly one continuous gap on one side.
    Every sprite inside a frame is mirrored across the axis perpendicular to that
    side, translated straight outward until flush with the frame exterior, and
    recolored to the frame color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = _monochrome_components(g, bg)

    frames: list[dict] = []
    frame_ids: set[int] = set()
    for idx, comp in enumerate(comps):
        info = _frame_info(comp)
        if info is not None:
            info["component_id"] = idx
            frames.append(info)
            frame_ids.add(idx)

    assert frames, "expected at least one frame"
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))

    out = g.copy()
    h, w = g.shape

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        side = frame["gap_side"]
        frame_color = int(frame["color"])

        for idx, comp in enumerate(comps):
            if idx in frame_ids:
                continue
            coords = comp["coords"]
            if not all(r0 < r < r1 and c0 < c < c1 for r, c in coords):
                continue

            sr0, sc0, sr1, sc1 = comp["bbox"]
            mask = _component_mask(comp)
            if side in ("left", "right"):
                mask = np.fliplr(mask)
            else:
                mask = np.flipud(mask)

            ph, pw = mask.shape
            if side == "right":
                tr, tc = sr0, c1 + 1
            elif side == "left":
                tr, tc = sr0, c0 - pw
            elif side == "bottom":
                tr, tc = r1 + 1, sc0
            else:
                tr, tc = r0 - ph, sc0

            assert 0 <= tr and 0 <= tc and tr + ph <= h and tc + pw <= w, "projection out of bounds"
            rows, cols = np.nonzero(mask)
            out[tr + rows, tc + cols] = frame_color

    return out


# --------------- generator ---------------

BG = 0
CANVAS_H_RANGE = (24, 34)
CANVAS_W_RANGE = (24, 34)
FRAME_COUNT_RANGE = (1, 3)
SPRITES_PER_FRAME = (1, 2)
FRAME_SIZE_MIN = 7
FRAME_SIZE_MAX = 13
SPRITE_SIZE_MAX = 4
GENERATION_ATTEMPTS = 500
FRAME_BUILD_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 16


def _random_connected_mask(height: int, width: int, flip_kind: str) -> np.ndarray | None:
    area = height * width
    min_fill = max(3, (area + 1) // 2)
    max_fill = area - 1
    if max_fill < min_fill:
        return None

    for _ in range(100):
        target = random.randint(min_fill, max_fill)
        mask = np.zeros((height, width), dtype=bool)
        start = (random.randrange(height), random.randrange(width))
        mask[start] = True
        frontier = set()

        def add_frontier(cell: tuple[int, int]) -> None:
            r, c = cell
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))

        add_frontier(start)
        while int(mask.sum()) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.discard(cell)
            mask[cell] = True
            add_frontier(cell)

        if int(mask.sum()) != target:
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if flip_kind == "lr" and np.array_equal(mask, np.fliplr(mask)):
            continue
        if flip_kind == "ud" and np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    return None


def _sample_frame_size(side: str, sprite_count: int) -> tuple[int, int]:
    min_h = FRAME_SIZE_MIN
    min_w = FRAME_SIZE_MIN
    if side in ("left", "right") and sprite_count == 2:
        min_h = 9
    if side in ("top", "bottom") and sprite_count == 2:
        min_w = 9
    return random.randint(min_h, FRAME_SIZE_MAX), random.randint(min_w, FRAME_SIZE_MAX)


def _intervals_separated(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a1 + 1 < b0 or b1 + 1 < a0


def _place_sprites_inside_frame(
    frame_h: int, frame_w: int, side: str, sprite_colors: list[int]
) -> list[dict] | None:
    core_r0, core_c0 = 2, 2
    core_h = frame_h - 4
    core_w = frame_w - 4
    flip_kind = "lr" if side in ("left", "right") else "ud"

    for _ in range(FRAME_BUILD_ATTEMPTS):
        sprites: list[dict] = []
        ok = True
        for color in sprite_colors:
            max_h = min(SPRITE_SIZE_MAX, core_h)
            max_w = min(SPRITE_SIZE_MAX, core_w)
            if max_h < 2 or max_w < 2:
                return None

            sh = random.randint(2, max_h)
            sw = random.randint(2, max_w)
            mask = _random_connected_mask(sh, sw, flip_kind)
            if mask is None:
                ok = False
                break

            placed = False
            row_positions = list(range(core_r0, core_r0 + core_h - sh + 1))
            col_positions = list(range(core_c0, core_c0 + core_w - sw + 1))
            random.shuffle(row_positions)
            random.shuffle(col_positions)

            for r in row_positions:
                for c in col_positions:
                    if side in ("left", "right"):
                        conflict = any(
                            not _intervals_separated(r, r + sh - 1, spr["r"], spr["r"] + spr["h"] - 1)
                            for spr in sprites
                        )
                    else:
                        conflict = any(
                            not _intervals_separated(c, c + sw - 1, spr["c"], spr["c"] + spr["w"] - 1)
                            for spr in sprites
                        )
                    if conflict:
                        continue
                    sprites.append(
                        {
                            "color": int(color),
                            "r": int(r),
                            "c": int(c),
                            "h": int(sh),
                            "w": int(sw),
                            "mask": mask,
                        }
                    )
                    placed = True
                    break
                if placed:
                    break

            if not placed:
                ok = False
                break

        if ok:
            return sprites
    return None


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    rows, cols = np.nonzero(mask)
    canvas[r0 + rows, c0 + cols] = int(color)


def _draw_frame_border(canvas: np.ndarray, fr: int, fc: int, frame_h: int, frame_w: int, color: int, side: str) -> None:
    canvas[fr, fc : fc + frame_w] = color
    canvas[fr + frame_h - 1, fc : fc + frame_w] = color
    canvas[fr : fr + frame_h, fc] = color
    canvas[fr : fr + frame_h, fc + frame_w - 1] = color

    side_len = frame_w if side in ("top", "bottom") else frame_h
    gap_len = random.randint(1, side_len - 4)
    gap_start = random.randint(1, side_len - gap_len - 1)
    gap_slice = slice(gap_start, gap_start + gap_len)

    if side == "top":
        canvas[fr, fc + gap_slice.start : fc + gap_slice.stop] = BG
    elif side == "bottom":
        canvas[fr + frame_h - 1, fc + gap_slice.start : fc + gap_slice.stop] = BG
    elif side == "left":
        canvas[fr + gap_slice.start : fr + gap_slice.stop, fc] = BG
    else:
        canvas[fr + gap_slice.start : fr + gap_slice.stop, fc + frame_w - 1] = BG


def _build_frame_patch(frame_color: int, sprite_colors: list[int], side: str) -> dict | None:
    for _ in range(FRAME_BUILD_ATTEMPTS):
        frame_h, frame_w = _sample_frame_size(side, len(sprite_colors))
        sprites = _place_sprites_inside_frame(frame_h, frame_w, side, sprite_colors)
        if sprites is None:
            continue

        max_proj_h = max(spr["h"] for spr in sprites)
        max_proj_w = max(spr["w"] for spr in sprites)

        if side == "right":
            patch_h, patch_w = frame_h, frame_w + max_proj_w
            fr, fc = 0, 0
        elif side == "left":
            patch_h, patch_w = frame_h, frame_w + max_proj_w
            fr, fc = 0, max_proj_w
        elif side == "bottom":
            patch_h, patch_w = frame_h + max_proj_h, frame_w
            fr, fc = 0, 0
        else:
            patch_h, patch_w = frame_h + max_proj_h, frame_w
            fr, fc = max_proj_h, 0

        inp = np.full((patch_h, patch_w), BG, dtype=int)
        out = np.full((patch_h, patch_w), BG, dtype=int)
        _draw_frame_border(inp, fr, fc, frame_h, frame_w, frame_color, side)
        out[:] = inp

        for spr in sprites:
            _paint_mask(out, spr["mask"], fr + spr["r"], fc + spr["c"], spr["color"])
            _paint_mask(inp, spr["mask"], fr + spr["r"], fc + spr["c"], spr["color"])

            proj = np.fliplr(spr["mask"]) if side in ("left", "right") else np.flipud(spr["mask"])
            if side == "right":
                tr, tc = fr + spr["r"], fc + frame_w
            elif side == "left":
                tr, tc = fr + spr["r"], fc - spr["w"]
            elif side == "bottom":
                tr, tc = fr + frame_h, fc + spr["c"]
            else:
                tr, tc = fr - spr["h"], fc + spr["c"]
            _paint_mask(out, proj, tr, tc, frame_color)

        return {"input": inp, "output": out}
    return None


def _paste_visible(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    mask = patch != bg
    canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]


def _find_free_rect(occupied: np.ndarray, shape: tuple[int, int]) -> tuple[int, int] | None:
    ph, pw = shape
    h, w = occupied.shape
    if ph > h or pw > w:
        return None

    positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(positions)
    for r, c in positions:
        if not occupied[r : r + ph, c : c + pw].any():
            return r, c
    return None


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    sprite_counts = [random.randint(*SPRITES_PER_FRAME) for _ in range(frame_count)]
    total_colors = frame_count + sum(sprite_counts)
    if total_colors > 9:
        return None

    palette = random.sample(list(range(1, 10)), total_colors)
    frame_colors = palette[:frame_count]
    cursor = frame_count

    specs: list[dict] = []
    for frame_color, sprite_count in zip(frame_colors, sprite_counts):
        sprite_colors = palette[cursor : cursor + sprite_count]
        cursor += sprite_count
        side = random.choice(SIDES)
        spec = _build_frame_patch(frame_color, sprite_colors, side)
        if spec is None:
            return None
        specs.append(spec)

    max_patch_h = max(spec["input"].shape[0] for spec in specs)
    max_patch_w = max(spec["input"].shape[1] for spec in specs)
    total_area = sum(int(spec["input"].shape[0] * spec["input"].shape[1]) for spec in specs)

    candidates = [
        (h, w)
        for h in range(max(CANVAS_H_RANGE[0], max_patch_h), CANVAS_H_RANGE[1] + 1)
        for w in range(max(CANVAS_W_RANGE[0], max_patch_w), CANVAS_W_RANGE[1] + 1)
        if h * w >= total_area + 40 * frame_count
    ]
    if not candidates:
        return None

    order = specs[:]
    random.shuffle(order)
    order.sort(key=lambda item: item["input"].shape[0] * item["input"].shape[1], reverse=True)

    for _ in range(PLACEMENT_ATTEMPTS):
        h, w = random.choice(candidates)
        inp = np.full((h, w), BG, dtype=int)
        out = np.full((h, w), BG, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        ok = True

        for spec in order:
            loc = _find_free_rect(occupied, spec["input"].shape)
            if loc is None:
                ok = False
                break
            r0, c0 = loc
            ph, pw = spec["input"].shape
            occupied[r0 : r0 + ph, c0 : c0 + pw] = True
            _paste_visible(inp, spec["input"], r0, c0, BG)
            _paste_visible(out, spec["output"], r0, c0, BG)

        if ok:
            try:
                if np.array_equal(main(inp), out):
                    return inp
            except Exception:
                pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
