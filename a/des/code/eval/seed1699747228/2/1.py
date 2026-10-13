from __future__ import annotations

import random
from collections import deque

import numpy as np


DIRECTIONS = ("up", "right", "down", "left")
DIR_INDEX = {name: idx for idx, name in enumerate(DIRECTIONS)}
VERTICAL = {"up", "down"}

BODY_LENGTH_CHOICES = (2, 3)
SCALE_CHOICES = (1, 2)
FRAME_COUNT_RANGE = (2, 4)
GRID_H_RANGE = (22, 32)
GRID_W_RANGE = (22, 32)
EXTRA_MARGIN_RANGE = (0, 4)
FRAME_PAD = 2
SPRITE_PAD = 1
MAX_PLACEMENT_TRIES = 200
MAX_GENERATION_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(flat, minlength=10).argmax())


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            color = int(g[r0, c0])
            if color == bg or seen[r0, c0]:
                continue
            seen[r0, c0] = True
            queue = deque([(r0, c0)])
            coords: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            rr0, rr1 = min(rows), max(rows)
            cc0, cc1 = min(cols), max(cols)
            mask = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
            for r, c in coords:
                mask[r - rr0, c - cc0] = True
            out.append(
                {
                    "color": color,
                    "bbox": (rr0, cc0, rr1, cc1),
                    "mask": mask,
                }
            )
    return out


def _frame_mask(height: int, width: int, opening: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    if opening == "up":
        mask[0, 1:-1] = False
    elif opening == "down":
        mask[-1, 1:-1] = False
    elif opening == "left":
        mask[1:-1, 0] = False
    else:
        mask[1:-1, -1] = False
    return mask


def _frame_opening(mask: np.ndarray) -> str | None:
    h, w = map(int, mask.shape)
    if h < 5 or w < 5:
        return None
    for opening in DIRECTIONS:
        if np.array_equal(mask, _frame_mask(h, w, opening)):
            return opening
    return None


def _up_arrow_mask(body_length: int) -> np.ndarray:
    mask = np.ones((body_length + 1, 3), dtype=bool)
    mask[0, :] = False
    mask[0, 1] = True
    return mask


def _rotate_to(mask: np.ndarray, src_dir: str, dst_dir: str) -> np.ndarray:
    k = (DIR_INDEX[src_dir] - DIR_INDEX[dst_dir]) % 4
    return np.rot90(mask, k)


def _arrow_mask(direction: str, body_length: int) -> np.ndarray:
    return _rotate_to(_up_arrow_mask(body_length), "up", direction)


def _arrow_direction(mask: np.ndarray) -> str | None:
    h, w = map(int, mask.shape)
    if w == 3 and h >= 3:
        if bool(mask[0, 1]) and int(mask[0].sum()) == 1 and bool(mask[1:, :].all()):
            return "up"
        if bool(mask[-1, 1]) and int(mask[-1].sum()) == 1 and bool(mask[:-1, :].all()):
            return "down"
    if h == 3 and w >= 3:
        if bool(mask[1, 0]) and int(mask[:, 0].sum()) == 1 and bool(mask[:, 1:].all()):
            return "left"
        if bool(mask[1, -1]) and int(mask[:, -1].sum()) == 1 and bool(mask[:, :-1].all()):
            return "right"
    return None


def _scale_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = map(int, mask.shape)
    dst_h, dst_w = map(int, target_shape)
    row_idx = (np.arange(dst_h) * src_h) // dst_h
    col_idx = (np.arange(dst_w) * src_w) // dst_w
    return mask[row_idx][:, col_idx]


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = map(int, mask.shape)
    sub = grid[top : top + h, left : left + w]
    sub[mask] = int(color)


def _build_output(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for frame in frames:
        top = int(frame["top"])
        left = int(frame["left"])
        outer_h = int(frame["outer_h"])
        outer_w = int(frame["outer_w"])
        color = int(frame["color"])
        opening = str(frame["opening"])
        body_length = int(frame["body_length"])
        scale = int(frame["scale"])

        _paint_mask(out, top, left, _frame_mask(outer_h, outer_w, opening), color)

        inner_h = outer_h - 2
        inner_w = outer_w - 2
        arrow = _arrow_mask(opening, body_length)
        target_shape = (arrow.shape[0] * scale, arrow.shape[1] * scale)
        scaled = _scale_mask(arrow, target_shape)
        place_r = top + 1 + (inner_h - scaled.shape[0]) // 2
        place_c = left + 1 + (inner_w - scaled.shape[1]) // 2
        _paint_mask(out, place_r, place_c, scaled, color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    For each monochrome U-frame, rotate the matching same-color arrow sprite so it
    points toward the opening, scale it to the frame's inner cross dimension, and
    place it centered inside the frame. Remove all external sprites.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    comps = _components(g, bg)

    frames: dict[int, dict] = {}
    sprites: dict[int, dict] = {}
    for comp in comps:
        opening = _frame_opening(comp["mask"])
        if opening is not None:
            rr0, cc0, rr1, cc1 = map(int, comp["bbox"])
            frames[int(comp["color"])] = {
                "color": int(comp["color"]),
                "bbox": (rr0, cc0, rr1, cc1),
                "opening": opening,
            }
            continue
        direction = _arrow_direction(comp["mask"])
        if direction is not None:
            sprites[int(comp["color"])] = {
                "color": int(comp["color"]),
                "mask": comp["mask"],
                "direction": direction,
            }

    assert frames and len(frames) == len(sprites)

    out = np.full(g.shape, bg, dtype=int)
    for color, frame in frames.items():
        assert color in sprites
        rr0, cc0, rr1, cc1 = map(int, frame["bbox"])
        outer_h = rr1 - rr0 + 1
        outer_w = cc1 - cc0 + 1
        opening = str(frame["opening"])
        _paint_mask(out, rr0, cc0, _frame_mask(outer_h, outer_w, opening), color)

        sprite = sprites[color]
        rotated = _rotate_to(sprite["mask"], str(sprite["direction"]), opening)
        inner_h = outer_h - 2
        inner_w = outer_w - 2
        if opening in VERTICAL:
            scale = inner_w / rotated.shape[1]
        else:
            scale = inner_h / rotated.shape[0]
        target_h = max(1, int(round(rotated.shape[0] * scale)))
        target_w = max(1, int(round(rotated.shape[1] * scale)))
        assert target_h <= inner_h and target_w <= inner_w
        scaled = _scale_mask(rotated, (target_h, target_w))
        place_r = rr0 + 1 + (inner_h - target_h) // 2
        place_c = cc0 + 1 + (inner_w - target_w) // 2
        _paint_mask(out, place_r, place_c, scaled, color)

    return out


def _sample_frame_specs() -> list[dict]:
    n_frames = random.randint(*FRAME_COUNT_RANGE)
    colors = random.sample(range(1, 10), n_frames)
    frames: list[dict] = []
    for color in colors:
        opening = random.choice(DIRECTIONS)
        sprite_dir = random.choice(DIRECTIONS)
        body_length = random.choice(BODY_LENGTH_CHOICES)
        scale = random.choice(SCALE_CHOICES)
        extra = random.randint(*EXTRA_MARGIN_RANGE)

        if opening in VERTICAL:
            inner_w = 3 * scale
            arrow_h = (body_length + 1) * scale
            inner_h = arrow_h + extra
        else:
            inner_h = 3 * scale
            arrow_w = (body_length + 1) * scale
            inner_w = arrow_w + extra

        frames.append(
            {
                "color": color,
                "opening": opening,
                "sprite_dir": sprite_dir,
                "body_length": body_length,
                "scale": scale,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
            }
        )
    return frames


def _can_place_rect(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> bool:
    h, w = map(int, blocked.shape)
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not bool(blocked[r0:r1, c0:c1].any())


def _mark_rect(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> None:
    h, w = map(int, blocked.shape)
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    blocked[r0:r1, c0:c1] = True


def _place_frames(shape: tuple[int, int], frames: list[dict]) -> bool:
    blocked = np.zeros(shape, dtype=bool)
    order = sorted(range(len(frames)), key=lambda i: frames[i]["outer_h"] * frames[i]["outer_w"], reverse=True)
    for idx in order:
        frame = frames[idx]
        outer_h = int(frame["outer_h"])
        outer_w = int(frame["outer_w"])
        placed = False
        for _ in range(MAX_PLACEMENT_TRIES):
            top = random.randint(0, shape[0] - outer_h)
            left = random.randint(0, shape[1] - outer_w)
            if not _can_place_rect(blocked, top, left, outer_h, outer_w, FRAME_PAD):
                continue
            frame["top"] = top
            frame["left"] = left
            _mark_rect(blocked, top, left, outer_h, outer_w, FRAME_PAD)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_input_sprites(inp: np.ndarray, frames: list[dict]) -> bool:
    blocked = np.zeros(inp.shape, dtype=bool)
    for frame in frames:
        _mark_rect(
            blocked,
            int(frame["top"]),
            int(frame["left"]),
            int(frame["outer_h"]),
            int(frame["outer_w"]),
            SPRITE_PAD,
        )

    order = list(frames)
    random.shuffle(order)
    for frame in order:
        mask = _arrow_mask(str(frame["sprite_dir"]), int(frame["body_length"]))
        height, width = map(int, mask.shape)
        placed = False
        for _ in range(MAX_PLACEMENT_TRIES):
            top = random.randint(0, inp.shape[0] - height)
            left = random.randint(0, inp.shape[1] - width)
            if not _can_place_rect(blocked, top, left, height, width, SPRITE_PAD):
                continue
            _paint_mask(inp, top, left, mask, int(frame["color"]))
            _mark_rect(blocked, top, left, height, width, SPRITE_PAD)
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    frames = _sample_frame_specs()
    min_h = max(int(frame["outer_h"]) for frame in frames)
    min_w = max(int(frame["outer_w"]) for frame in frames)

    lo_h = max(GRID_H_RANGE[0], min_h + 2)
    lo_w = max(GRID_W_RANGE[0], min_w + 2)
    if lo_h > GRID_H_RANGE[1] or lo_w > GRID_W_RANGE[1]:
        return None

    shape = (
        random.randint(lo_h, GRID_H_RANGE[1]),
        random.randint(lo_w, GRID_W_RANGE[1]),
    )
    if not _place_frames(shape, frames):
        return None

    inp = np.zeros(shape, dtype=int)
    for frame in frames:
        _paint_mask(
            inp,
            int(frame["top"]),
            int(frame["left"]),
            _frame_mask(int(frame["outer_h"]), int(frame["outer_w"]), str(frame["opening"])),
            int(frame["color"]),
        )
    if not _place_input_sprites(inp, frames):
        return None

    out = _build_output(shape, frames)
    try:
        if np.array_equal(main(inp), out):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
