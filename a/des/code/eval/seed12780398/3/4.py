from __future__ import annotations

import random
from collections import Counter

import numpy as np


FRAME_COLOR = 5
DIR_VECS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
DIR_ORDER_CCW = ["up", "left", "down", "right"]
SPRITE_TEMPLATES = {
    2: [
        np.array(
            [
                [0, 1],
                [1, 1],
                [1, 1],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 0],
                [1, 1],
                [1, 1],
            ],
            dtype=bool,
        ),
    ],
    3: [
        np.array(
            [
                [0, 1, 0],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=bool,
        ),
    ],
}


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


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
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _rotate_dir_ccw(direction: str, k: int) -> str:
    idx = DIR_ORDER_CCW.index(direction)
    return DIR_ORDER_CCW[(idx + k) % 4]


def _rotation_steps_ccw(src: str, dst: str) -> int:
    for k in range(4):
        if _rotate_dir_ccw(src, k) == dst:
            return k
    raise AssertionError("unreachable")


def _sorted_line(coords: list[tuple[int, int]], direction: str) -> list[tuple[int, int]]:
    if direction in ("up", "down"):
        return sorted(coords, key=lambda p: (p[1], p[0]))
    return sorted(coords, key=lambda p: (p[0], p[1]))


def _gap_edge_coords(mask: np.ndarray, direction: str) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    if direction == "up":
        edge = [(r, c) for r, c in pts if r == mask.shape[0] - 1]
    elif direction == "down":
        edge = [(r, c) for r, c in pts if r == 0]
    elif direction == "left":
        edge = [(r, c) for r, c in pts if c == mask.shape[1] - 1]
    else:
        edge = [(r, c) for r, c in pts if c == 0]
    return _sorted_line(edge, direction)


def _place_sprite_at_gap(
    mask: np.ndarray, direction: str, gap_coords: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    edge = _gap_edge_coords(mask, direction)
    gap = _sorted_line(gap_coords, direction)
    assert len(edge) == len(gap) and edge, "sprite edge must match the gap length"
    dr = gap[0][0] - edge[0][0]
    dc = gap[0][1] - edge[0][1]
    for (er, ec), (gr, gc) in zip(edge, gap):
        assert er + dr == gr and ec + dc == gc, "gap alignment must be rigid"
    return [(int(r + dr), int(c + dc)) for r, c in np.argwhere(mask)]


def _parse_frame_component(coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(mask & ~border):
        return None

    missing = np.argwhere(border & ~mask)
    if missing.shape[0] not in (2, 3):
        return None

    if np.all(missing[:, 0] == 0):
        direction = "up"
        idxs = sorted(int(x) for x in missing[:, 1])
        edge_limit = w - 1
        gap_coords = [(r0, c0 + i) for i in idxs]
    elif np.all(missing[:, 0] == h - 1):
        direction = "down"
        idxs = sorted(int(x) for x in missing[:, 1])
        edge_limit = w - 1
        gap_coords = [(r1, c0 + i) for i in idxs]
    elif np.all(missing[:, 1] == 0):
        direction = "left"
        idxs = sorted(int(x) for x in missing[:, 0])
        edge_limit = h - 1
        gap_coords = [(r0 + i, c0) for i in idxs]
    elif np.all(missing[:, 1] == w - 1):
        direction = "right"
        idxs = sorted(int(x) for x in missing[:, 0])
        edge_limit = h - 1
        gap_coords = [(r0 + i, c1) for i in idxs]
    else:
        return None

    if idxs[0] == 0 or idxs[-1] == edge_limit or idxs[-1] - idxs[0] + 1 != len(idxs):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap_direction": direction,
        "gap_coords": gap_coords,
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != bg):
        for coords in _components(grid == color):
            parsed = _parse_frame_component(coords)
            if parsed is None:
                continue
            parsed["color"] = int(color)
            frames.append(parsed)
    frames.sort(key=lambda frame: frame["bbox"])
    assert frames, "expected at least one frame"
    return frames


def _find_arrow(grid: np.ndarray, frame: dict, bg: int) -> dict:
    r0, c0, r1, c1 = frame["bbox"]
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (inner != bg) & (inner != frame["color"])
    comps = _components(mask)
    assert len(comps) == 1, "each frame should contain exactly one internal marker"
    coords_local = comps[0]
    assert coords_local.shape[0] == 3, "arrow must use three pixels"

    colors = {int(inner[r, c]) for r, c in coords_local}
    assert len(colors) == 1, "arrow must be monochrome"
    color = int(next(iter(colors)))

    coords = [(int(r0 + 1 + r), int(c0 + 1 + c)) for r, c in coords_local]
    coord_set = set(coords)
    degrees = {
        pt: sum((nbr in coord_set) for nbr in _neighbors4(*pt))
        for pt in coord_set
    }
    elbow = [pt for pt, deg in degrees.items() if deg == 2]
    ends = [pt for pt, deg in degrees.items() if deg == 1]
    assert len(elbow) == 1 and len(ends) == 2, "arrow must be an L triomino"

    center = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0], dtype=float)
    center_end = min(
        ends,
        key=lambda pt: (pt[0] - center[0]) ** 2 + (pt[1] - center[1]) ** 2,
    )
    tip = ends[1] if ends[0] == center_end else ends[0]
    dr = tip[0] - elbow[0][0]
    dc = tip[1] - elbow[0][1]
    direction = None
    for name, vec in DIR_VECS.items():
        if vec == (dr, dc):
            direction = name
            break
    assert direction is not None, "arrow direction must be cardinal"

    return {
        "color": color,
        "coords": coords,
        "direction": direction,
    }


def _match_sprite(mask: np.ndarray) -> dict:
    for gap_len, variants in SPRITE_TEMPLATES.items():
        for variant_idx, base in enumerate(variants):
            for k in range(4):
                cand = np.rot90(base, k)
                if cand.shape == mask.shape and np.array_equal(cand, mask):
                    return {
                        "gap_len": int(gap_len),
                        "variant": int(variant_idx),
                        "direction": _rotate_dir_ccw("up", k),
                    }
    raise AssertionError("unrecognized sprite shape")


def _component_inside_bbox(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(
        np.all((coords[:, 0] >= r0) & (coords[:, 0] <= r1) & (coords[:, 1] >= c0) & (coords[:, 1] <= c1))
    )


def _find_external_sprites(
    grid: np.ndarray, colors: set[int], frames: list[dict]
) -> dict[int, dict]:
    sprites: dict[int, dict] = {}
    for color in colors:
        comps = _components(grid == color)
        choices = []
        for coords in comps:
            if coords.shape[0] <= 3:
                continue
            if any(_component_inside_bbox(coords, frame["bbox"]) for frame in frames):
                continue
            mask = _tight_mask(coords)
            info = _match_sprite(mask)
            choices.append(
                {
                    "coords": [(int(r), int(c)) for r, c in coords],
                    "mask": mask,
                    "direction": info["direction"],
                    "gap_len": info["gap_len"],
                }
            )
        assert len(choices) == 1, "each arrow color must have exactly one outside sprite"
        sprites[int(color)] = choices[0]
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each frame's internal arrow to the same-colored outside sprite, rotate it to the
    arrow direction, and move it into the frame gap while erasing the arrow.
    """
    g = np.asarray(grid, dtype=int)
    bg = _majority_color(g)
    frames = _find_frames(g, bg)

    out = g.copy()
    arrows_by_color: dict[int, dict] = {}
    for frame in frames:
        arrow = _find_arrow(g, frame, bg)
        assert arrow["color"] not in arrows_by_color, "arrow colors must be unique"
        arrows_by_color[arrow["color"]] = {"frame": frame, "arrow": arrow}
        for r, c in arrow["coords"]:
            out[r, c] = bg

    sprites = _find_external_sprites(g, set(arrows_by_color), frames)
    for color, sprite in sprites.items():
        frame = arrows_by_color[color]["frame"]
        arrow = arrows_by_color[color]["arrow"]
        for r, c in sprite["coords"]:
            out[r, c] = bg

        k = _rotation_steps_ccw(sprite["direction"], arrow["direction"])
        rotated = np.rot90(sprite["mask"], k)
        placed = _place_sprite_at_gap(rotated, frame["gap_direction"], frame["gap_coords"])
        assert len(frame["gap_coords"]) == sprite["gap_len"], "sprite width must match its frame gap"
        for r, c in placed:
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1], "placed sprite must stay in bounds"
            assert out[r, c] == bg, "placed sprite must only cover background"
        for r, c in placed:
            out[r, c] = color

    return out


def _dilate_points(mask: np.ndarray, points: list[tuple[int, int]], pad: int = 1) -> None:
    h, w = mask.shape
    for r, c in points:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                mask[rr, cc] = True


def _all_border_coords(h: int, w: int, gap_direction: str, gap_start: int, gap_len: int) -> list[tuple[int, int]]:
    coords = [(0, c) for c in range(w)]
    coords += [(h - 1, c) for c in range(w)]
    coords += [(r, 0) for r in range(1, h - 1)]
    coords += [(r, w - 1) for r in range(1, h - 1)]
    gap = set(_local_gap_coords(h, w, gap_direction, gap_start, gap_len))
    return sorted(set(coords) - gap)


def _local_gap_coords(h: int, w: int, direction: str, start: int, gap_len: int) -> list[tuple[int, int]]:
    if direction == "up":
        return [(0, c) for c in range(start, start + gap_len)]
    if direction == "down":
        return [(h - 1, c) for c in range(start, start + gap_len)]
    if direction == "left":
        return [(r, 0) for r in range(start, start + gap_len)]
    return [(r, w - 1) for r in range(start, start + gap_len)]


def _valid_gap_starts(h: int, w: int, direction: str, gap_len: int) -> list[int]:
    span = w if direction in ("up", "down") else h
    return list(range(1, span - gap_len))


def _arrow_local_coords(direction: str, mirror: int, center_r: int, center_c: int) -> list[tuple[int, int]]:
    dr, dc = DIR_VECS[direction]
    if direction in ("up", "down"):
        shafts = [(0, 1), (0, -1)]
    else:
        shafts = [(1, 0), (-1, 0)]
    sr, sc = shafts[mirror]
    return [
        (center_r, center_c),
        (center_r + sr, center_c + sc),
        (center_r + sr + dr, center_c + sc + dc),
    ]


def _frame_reserved_box(spec: dict) -> tuple[int, int, int, int]:
    h, w = spec["height"], spec["width"]
    occupied = [(r, c) for r in range(h) for c in range(w)]
    occupied += spec["final_local_coords"]
    rows = [r for r, _ in occupied]
    cols = [c for _, c in occupied]
    return min(rows) - 1, min(cols) - 1, max(rows) + 1, max(cols) + 1


def _can_place_reserved(blocked: np.ndarray, bbox: tuple[int, int, int, int], top: int, left: int) -> bool:
    br0, bc0, br1, bc1 = bbox
    gr0, gc0, gr1, gc1 = top + br0, left + bc0, top + br1, left + bc1
    h, w = blocked.shape
    if gr0 < 0 or gc0 < 0 or gr1 >= h or gc1 >= w:
        return False
    return not bool(np.any(blocked[gr0 : gr1 + 1, gc0 : gc1 + 1]))


def _mark_reserved(blocked: np.ndarray, bbox: tuple[int, int, int, int], top: int, left: int) -> None:
    br0, bc0, br1, bc1 = bbox
    gr0, gc0, gr1, gc1 = top + br0, left + bc0, top + br1, left + bc1
    blocked[gr0 : gr1 + 1, gc0 : gc1 + 1] = True


def _candidate_positions(blocked: np.ndarray, patch_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = blocked.shape
    ph, pw = patch_mask.shape
    fg = [(int(r), int(c)) for r, c in np.argwhere(patch_mask)]
    positions: list[tuple[int, int]] = []
    for top in range(h - ph + 1):
        for left in range(w - pw + 1):
            if all(not blocked[top + r, left + c] for r, c in fg):
                positions.append((int(top), int(left)))
    return positions


def _draw_points(grid: np.ndarray, points: list[tuple[int, int]], color: int) -> None:
    for r, c in points:
        grid[r, c] = int(color)


def _build_expected(inp: np.ndarray, specs: list[dict], bg: int) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        for r, c in spec["arrow_global_coords"]:
            out[r, c] = bg
        for r, c in spec["sprite_input_global_coords"]:
            out[r, c] = bg
        _draw_points(out, spec["final_global_coords"], spec["arrow_color"])
    return out


def _attempt_generate() -> np.ndarray | None:
    n_frames = random.randint(1, 4)
    min_side = 18 + 4 * n_frames
    max_side = min_side + 8
    h = random.randint(min_side, max_side)
    w = random.randint(min_side, max_side)

    usable = [c for c in range(10) if c != FRAME_COLOR]
    bg = random.choice(usable)
    arrow_colors = random.sample([c for c in usable if c != bg], n_frames)

    specs: list[dict] = []
    for arrow_color in arrow_colors:
        direction = random.choice(list(DIR_VECS))
        gap_len = random.choice([2, 3])
        base = random.choice(SPRITE_TEMPLATES[gap_len]).copy()
        input_rotation = random.randrange(4)
        input_mask = np.rot90(base, input_rotation)
        final_rotation = _rotation_steps_ccw("up", direction)
        final_mask = np.rot90(base, final_rotation)

        frame_h = random.choice([7, 9])
        frame_w = random.choice([7, 9])
        gap_start = random.choice(_valid_gap_starts(frame_h, frame_w, direction, gap_len))
        gap_local = _local_gap_coords(frame_h, frame_w, direction, gap_start, gap_len)
        final_local = _place_sprite_at_gap(final_mask, direction, gap_local)

        center_r, center_c = frame_h // 2, frame_w // 2
        arrow_local = _arrow_local_coords(direction, random.randrange(2), center_r, center_c)

        specs.append(
            {
                "arrow_color": int(arrow_color),
                "direction": direction,
                "gap_len": int(gap_len),
                "height": int(frame_h),
                "width": int(frame_w),
                "gap_local_coords": gap_local,
                "border_local_coords": _all_border_coords(frame_h, frame_w, direction, gap_start, gap_len),
                "arrow_local_coords": arrow_local,
                "input_sprite_mask": input_mask,
                "final_local_coords": final_local,
            }
        )

    blocked = np.zeros((h, w), dtype=bool)
    order = list(range(n_frames))
    random.shuffle(order)
    order.sort(key=lambda idx: (_frame_reserved_box(specs[idx])[2] - _frame_reserved_box(specs[idx])[0]), reverse=True)
    for idx in order:
        spec = specs[idx]
        reserve = _frame_reserved_box(spec)
        positions = [
            (top, left)
            for top in range(h)
            for left in range(w)
            if _can_place_reserved(blocked, reserve, top, left)
        ]
        if not positions:
            return None
        top, left = random.choice(positions)
        spec["frame_top"] = int(top)
        spec["frame_left"] = int(left)
        spec["gap_global_coords"] = [(top + r, left + c) for r, c in spec["gap_local_coords"]]
        spec["arrow_global_coords"] = [(top + r, left + c) for r, c in spec["arrow_local_coords"]]
        spec["final_global_coords"] = [(top + r, left + c) for r, c in spec["final_local_coords"]]
        _mark_reserved(blocked, reserve, top, left)

    grid = np.full((h, w), bg, dtype=int)
    for spec in specs:
        _draw_points(
            grid,
            [(spec["frame_top"] + r, spec["frame_left"] + c) for r, c in spec["border_local_coords"]],
            FRAME_COLOR,
        )
        _draw_points(grid, spec["arrow_global_coords"], spec["arrow_color"])

    sprite_blocked = blocked.copy()
    for spec in specs:
        positions = _candidate_positions(sprite_blocked, spec["input_sprite_mask"])
        if not positions:
            return None
        top, left = random.choice(positions)
        spec["sprite_input_global_coords"] = [(top + r, left + c) for r, c in np.argwhere(spec["input_sprite_mask"])]
        _draw_points(grid, spec["sprite_input_global_coords"], spec["arrow_color"])
        _dilate_points(sprite_blocked, spec["sprite_input_global_coords"], pad=1)

    expected = _build_expected(grid, specs, bg)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        candidate = _attempt_generate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid instance")
