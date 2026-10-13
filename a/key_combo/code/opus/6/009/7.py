from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = list(range(1, 10))


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
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _sprite_patch(border_color: int, center_color: int) -> np.ndarray:
    patch = np.full((3, 3), int(border_color), dtype=int)
    patch[1, 1] = int(center_color)
    return patch


def _frame_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _direction_from_pair(
    sprite_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]
) -> tuple[int, int]:
    sr0, sc0, _sr1, _sc1 = sprite_bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    fh = fr1 - fr0 + 1
    fw = fc1 - fc0 + 1
    fr_center, fc_center = _frame_center(frame_bbox)
    sprite_center_r = sr0 + 1
    sprite_center_c = sc0 + 1
    if fw > fh:
        return (0, 1) if sprite_center_c <= fc_center else (0, -1)
    return (1, 0) if sprite_center_r <= fr_center else (-1, 0)


def _count_rooms(frame_patch: np.ndarray, bg: int) -> int:
    inner = frame_patch[1:-1, 1:-1]
    return len(_components_mask(inner == bg))


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        for coords in _components_mask(grid == color):
            bbox = _bbox_coords(coords)
            r0, c0, r1, c1 = bbox
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 4 or w < 4:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(patch[0, :] == color):
                continue
            if not np.all(patch[-1, :] == color):
                continue
            if not np.all(patch[:, 0] == color):
                continue
            if not np.all(patch[:, -1] == color):
                continue
            if not np.any(patch[1:-1, 1:-1] == bg):
                continue
            rooms = _count_rooms(patch, bg)
            if not 2 <= rooms <= 5:
                continue
            frames.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "patch": patch.copy(),
                    "rooms": int(rooms),
                }
            )
    return frames


def _find_sprites(grid: np.ndarray, bg: int, frame_colors: set[int]) -> list[dict]:
    sprites: list[dict] = []
    for coords in _components_mask(grid != bg):
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(patch != bg):
            continue
        border_mask = np.ones((3, 3), dtype=bool)
        border_mask[1, 1] = False
        border_colors = set(map(int, patch[border_mask]))
        if len(border_colors) != 1:
            continue
        border_color = int(next(iter(border_colors)))
        center_color = int(patch[1, 1])
        if border_color == center_color or center_color not in frame_colors:
            continue
        sprites.append(
            {
                "bbox": bbox,
                "patch": patch.copy(),
                "border_color": border_color,
                "center_color": center_color,
            }
        )
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat each matched 3x3 sprite as many times as the room count inside its same-colored divider frame.

    Input structure:
    - Background is black. The grid contains 2 or 3 monochrome divider frames: a rectangular outer border plus same-colored internal full-length divider lines that split the interior into enclosed background rooms.
    - The grid also contains 2 or 3 connected 3x3 sprites whose eight border pixels share one color and whose center pixel matches exactly one frame color.
    - Each frame is non-square, so its longer side gives a unique horizontal or vertical stamping axis.

    Rule:
    - Count the number N of 4-connected background rooms inside each frame.
    - Pair every sprite with the unique frame whose color matches the sprite center.
    - If the frame is wider than tall, stamp horizontally: sprites whose centers are at or left of the frame center move right, otherwise left. If the frame is taller than wide, stamp vertically: sprites at or above the frame center move down, otherwise up.
    - Keep the original sprite, then try to copy the full 3x3 sprite every 3 cells along that direction until there are N total copies. If a candidate 3x3 copy would overlap any non-background pixel, cancel that copy and every later copy.

    Key insight:
    - The frame is not the destination: its enclosed-room count becomes a numeric parameter that controls how many times the matching sprite motif can be stamped along the frame's major axis.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    frames = _find_frames(g, bg)
    assert 2 <= len(frames) <= 3
    frame_by_color = {frame["color"]: frame for frame in frames}
    assert len(frame_by_color) == len(frames)

    sprites = _find_sprites(g, bg, set(frame_by_color))
    assert len(sprites) == len(frames)

    for sprite in sorted(sprites, key=lambda d: d["bbox"]):
        frame = frame_by_color[sprite["center_color"]]
        dr, dc = _direction_from_pair(sprite["bbox"], frame["bbox"])
        sr0, sc0, _sr1, _sc1 = sprite["bbox"]
        copies = int(frame["rooms"])
        for step in range(1, copies):
            nr = sr0 + dr * 3 * step
            nc = sc0 + dc * 3 * step
            if nr < 0 or nc < 0 or nr + 3 > g.shape[0] or nc + 3 > g.shape[1]:
                break
            if np.any(out[nr : nr + 3, nc : nc + 3] != bg):
                break
            _paste(out, sprite["patch"], nr, nc)

    return out


def _factor_pairs(n_rooms: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for r in range(1, n_rooms + 1):
        if n_rooms % r == 0:
            pairs.append((r, n_rooms // r))
    return pairs


def _build_frame_patch(
    frame_color: int, n_rooms: int, orientation: str, compact: bool
) -> tuple[np.ndarray, int, int]:
    size_hi = 3 if compact else 4
    pairs = _factor_pairs(n_rooms)
    if orientation == "horizontal":
        pairs.sort(key=lambda rc: (rc[1] - rc[0], -rc[0]), reverse=True)
    else:
        pairs.sort(key=lambda rc: (rc[0] - rc[1], -rc[1]), reverse=True)
    pairs = pairs[: max(2, min(len(pairs), 3))]

    for _ in range(80):
        row_bands, col_bands = random.choice(pairs)
        row_sizes = [random.randint(2, size_hi) for _ in range(row_bands)]
        col_sizes = [random.randint(2, size_hi + (1 if orientation == "horizontal" else 0)) for _ in range(col_bands)]
        h = 2 + sum(row_sizes) + (row_bands - 1)
        w = 2 + sum(col_sizes) + (col_bands - 1)
        if orientation == "horizontal" and not (w > h):
            continue
        if orientation == "vertical" and not (h > w):
            continue
        if max(h, w) > 12:
            continue

        patch = np.zeros((h, w), dtype=int)
        patch[0, :] = frame_color
        patch[-1, :] = frame_color
        patch[:, 0] = frame_color
        patch[:, -1] = frame_color

        offset = 1 + row_sizes[0]
        for size in row_sizes[1:]:
            patch[offset, 1:-1] = frame_color
            offset += 1 + size

        offset = 1 + col_sizes[0]
        for size in col_sizes[1:]:
            patch[1:-1, offset] = frame_color
            offset += 1 + size

        if _count_rooms(patch, 0) == n_rooms:
            return patch, row_bands, col_bands
    raise RuntimeError("failed to build frame patch")


def _simulate_pair(
    frame_patch: np.ndarray,
    sprite_patch: np.ndarray,
    sprite_top: tuple[int, int],
    direction: tuple[int, int],
    rooms: int,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]], bool]:
    fh, fw = frame_patch.shape
    occupied = {
        (r, c)
        for r in range(fh)
        for c in range(fw)
        if int(frame_patch[r, c]) != 0
    }
    positions = [tuple(map(int, sprite_top))]
    for pr in range(3):
        for pc in range(3):
            occupied.add((sprite_top[0] + pr, sprite_top[1] + pc))

    stopped_early = False
    dr, dc = direction
    for step in range(1, rooms):
        nr = sprite_top[0] + dr * 3 * step
        nc = sprite_top[1] + dc * 3 * step
        candidate = {(nr + pr, nc + pc) for pr in range(3) for pc in range(3)}
        if occupied & candidate:
            stopped_early = True
            break
        occupied |= candidate
        positions.append((nr, nc))
    return positions, occupied, stopped_early


def _render_pair_scene(
    frame_patch: np.ndarray,
    sprite_patch: np.ndarray,
    sprite_top: tuple[int, int],
    positions: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]]:
    fh, fw = frame_patch.shape
    cells = [(r, c) for r in range(fh) for c in range(fw) if int(frame_patch[r, c]) != 0]
    for top in positions:
        for pr in range(3):
            for pc in range(3):
                cells.append((top[0] + pr, top[1] + pc))
    min_r = min(r for r, _ in cells)
    max_r = max(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_c = max(c for _, c in cells)
    shift_r = -min_r
    shift_c = -min_c

    h = max_r - min_r + 1
    w = max_c - min_c + 1
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)

    _paste(inp, frame_patch, shift_r, shift_c)
    _paste(out, frame_patch, shift_r, shift_c)

    sr = sprite_top[0] + shift_r
    sc = sprite_top[1] + shift_c
    _paste(inp, sprite_patch, sr, sc)

    for top in positions:
        _paste(out, sprite_patch, top[0] + shift_r, top[1] + shift_c)

    frame_bbox = (shift_r, shift_c, shift_r + fh - 1, shift_c + fw - 1)
    sprite_bbox = (sr, sc, sr + 2, sc + 2)
    return inp, out, frame_bbox, sprite_bbox


def _build_pair(
    frame_color: int,
    sprite_border: int,
    rooms: int,
    orientation: str,
    collision: bool,
    compact: bool,
) -> dict:
    frame_patch, _rows, _cols = _build_frame_patch(frame_color, rooms, orientation, compact)
    sprite_patch = _sprite_patch(sprite_border, frame_color)
    fh, fw = frame_patch.shape

    if orientation == "horizontal":
        dir_sign = random.choice([-1, 1])
        if collision:
            free_steps = random.randint(1, min(2, rooms - 2))
            sr = random.randint(0, max(0, fh - 3))
            if dir_sign == 1:
                sprite_top = (sr, -3 * free_steps - 4)
            else:
                sprite_top = (sr, fw + 1 + 3 * free_steps)
            direction = (0, dir_sign)
        else:
            sr = -4 if random.choice([True, False]) else fh + 1
            if dir_sign == 1:
                sc = random.choice([-5, -2])
            else:
                sc = random.choice([fw - 1, fw + 2])
            sprite_top = (sr, sc)
            direction = (0, dir_sign)
    else:
        dir_sign = random.choice([-1, 1])
        if collision:
            free_steps = random.randint(1, min(2, rooms - 2))
            sc = random.randint(0, max(0, fw - 3))
            if dir_sign == 1:
                sprite_top = (-3 * free_steps - 4, sc)
            else:
                sprite_top = (fh + 1 + 3 * free_steps, sc)
            direction = (dir_sign, 0)
        else:
            sc = -4 if random.choice([True, False]) else fw + 1
            if dir_sign == 1:
                sr = random.choice([-5, -2])
            else:
                sr = random.choice([fh - 1, fh + 2])
            sprite_top = (sr, sc)
            direction = (dir_sign, 0)

    positions, _occupied, stopped_early = _simulate_pair(
        frame_patch, sprite_patch, sprite_top, direction, rooms
    )
    if collision and not stopped_early:
        raise RuntimeError("collision pair did not stop early")
    if not collision and len(positions) != rooms:
        raise RuntimeError("free pair failed to complete all copies")

    inp, out, frame_bbox, sprite_bbox = _render_pair_scene(
        frame_patch, sprite_patch, sprite_top, positions
    )
    return {
        "input_patch": inp,
        "output_patch": out,
        "frame_bbox_local": frame_bbox,
        "sprite_bbox_local": sprite_bbox,
        "frame_color": frame_color,
        "sprite_border": sprite_border,
        "rooms": rooms,
        "copies_done": len(positions),
        "direction": direction,
        "orientation": orientation,
        "stopped_early": bool(stopped_early),
    }


def _plan_pairs() -> list[dict] | None:
    pair_count = random.choice([2, 3])
    compact = pair_count == 3

    orientations = ["horizontal", "vertical"]
    if pair_count == 3:
        orientations.append(random.choice(["horizontal", "vertical"]))
    random.shuffle(orientations)

    room_choices = [2, 3, 4, 5]
    rooms = [random.choice(room_choices) for _ in range(pair_count)]
    if len(set(rooms)) == 1:
        rooms[random.randrange(pair_count)] = random.choice([v for v in room_choices if v != rooms[0]])

    eligible_for_collision = [i for i, n in enumerate(rooms) if n >= 3]
    collision_index = None
    if eligible_for_collision and random.random() < 0.65:
        collision_index = random.choice(eligible_for_collision)

    colors = random.sample(PALETTE, pair_count * 2)
    frame_colors = colors[:pair_count]
    sprite_borders = colors[pair_count:]

    pairs: list[dict] = []
    for i in range(pair_count):
        try:
            pair = _build_pair(
                frame_color=frame_colors[i],
                sprite_border=sprite_borders[i],
                rooms=rooms[i],
                orientation=orientations[i],
                collision=i == collision_index,
                compact=compact,
            )
        except RuntimeError:
            return None
        pairs.append(pair)

    if collision_index is not None and not any(pair["stopped_early"] for pair in pairs):
        return None
    if not any(pair["copies_done"] == pair["rooms"] for pair in pairs):
        return None
    return pairs


def _pack_pairs(pairs: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    hs = [pair["output_patch"].shape[0] for pair in pairs]
    ws = [pair["output_patch"].shape[1] for pair in pairs]
    need_h = max(max(hs) + 2, 22)
    need_w = max(max(ws) + 2, 22)
    if len(pairs) == 3:
        need_h = max(need_h, 25)
        need_w = max(need_w, 25)

    for _ in range(80):
        h = random.randint(need_h, 30)
        w = random.randint(need_w, 30)
        order = sorted(
            range(len(pairs)),
            key=lambda i: pairs[i]["output_patch"].shape[0] * pairs[i]["output_patch"].shape[1],
            reverse=True,
        )
        occ = np.zeros((h, w), dtype=bool)
        placements: dict[int, tuple[int, int]] = {}
        ok = True

        for idx in order:
            ph, pw = pairs[idx]["output_patch"].shape
            candidates = [
                (r, c)
                for r in range(1, h - ph)
                for c in range(1, w - pw)
            ]
            random.shuffle(candidates)
            placed = False
            for r0, c0 in candidates:
                rr0 = max(0, r0 - 2)
                cc0 = max(0, c0 - 2)
                rr1 = min(h, r0 + ph + 2)
                cc1 = min(w, c0 + pw + 2)
                if occ[rr0:rr1, cc0:cc1].any():
                    continue
                occ[rr0:rr1, cc0:cc1] = True
                placements[idx] = (r0, c0)
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        inp = np.zeros((h, w), dtype=int)
        out = np.zeros((h, w), dtype=int)
        meta: list[dict] = []
        for idx, pair in enumerate(pairs):
            r0, c0 = placements[idx]
            _paste(inp, pair["input_patch"], r0, c0)
            _paste(out, pair["output_patch"], r0, c0)

            fr0, fc0, fr1, fc1 = pair["frame_bbox_local"]
            sr0, sc0, sr1, sc1 = pair["sprite_bbox_local"]
            meta.append(
                {
                    "frame_color": pair["frame_color"],
                    "sprite_border": pair["sprite_border"],
                    "rooms": pair["rooms"],
                    "copies_done": pair["copies_done"],
                    "direction": pair["direction"],
                    "frame_bbox": (r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1),
                    "sprite_bbox": (r0 + sr0, c0 + sc0, r0 + sr1, c0 + sc1),
                    "stopped_early": pair["stopped_early"],
                    "orientation": pair["orientation"],
                }
            )
        return inp, out, meta
    return None


def _validate_instance(inp: np.ndarray, expected: np.ndarray, meta: list[dict]) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False
    if len(meta) not in (2, 3):
        return False
    if len({m["frame_color"] for m in meta}) != len(meta):
        return False
    if len({m["sprite_border"] for m in meta}) != len(meta):
        return False
    if set(m["frame_color"] for m in meta) & set(m["sprite_border"] for m in meta):
        return False

    parsed_frames = _find_frames(inp, 0)
    if len(parsed_frames) != len(meta):
        return False
    parsed_by_color = {frame["color"]: frame for frame in parsed_frames}
    if set(parsed_by_color) != {m["frame_color"] for m in meta}:
        return False
    for item in meta:
        frame = parsed_by_color[item["frame_color"]]
        if frame["rooms"] != item["rooms"] or frame["bbox"] != item["frame_bbox"]:
            return False
        if _direction_from_pair(item["sprite_bbox"], item["frame_bbox"]) != item["direction"]:
            return False

    parsed_sprites = _find_sprites(inp, 0, set(parsed_by_color))
    if len(parsed_sprites) != len(meta):
        return False
    sprite_by_center = {sprite["center_color"]: sprite for sprite in parsed_sprites}
    for item in meta:
        sprite = sprite_by_center.get(item["frame_color"])
        if sprite is None:
            return False
        if sprite["bbox"] != item["sprite_bbox"]:
            return False
        if sprite["border_color"] != item["sprite_border"]:
            return False

    if not any(item["copies_done"] == item["rooms"] for item in meta):
        return False

    solved = main(inp)
    return np.array_equal(solved, expected)


def generate_input() -> np.ndarray:
    for _ in range(400):
        pairs = _plan_pairs()
        if pairs is None:
            continue
        packed = _pack_pairs(pairs)
        if packed is None:
            continue
        inp, expected, meta = packed
        if _validate_instance(inp, expected, meta):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
