import random
from collections import Counter

import numpy as np


SIDES = ("top", "right", "bottom", "left")
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400
PAIR_COUNT_CHOICES = (2, 3)
GAP_WIDTH_POOL = (3, 4, 5, 6)
OTHER_DIM_POOL = (2, 3, 4, 5, 6)
CANVAS_MIN = 18
CANVAS_MAX = 30


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            bbox = (int(r0), int(c0), int(r1), int(c1))
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox,
                    "mask": _component_mask(arr, bbox),
                }
            )
    return comps


def _false_runs(arr: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    i = 0
    while i < len(arr):
        if bool(arr[i]):
            i += 1
            continue
        j = i
        while j < len(arr) and not bool(arr[j]):
            j += 1
        runs.append((i, j - i))
        i = j
    return runs


def _frame_info(comp: dict) -> dict | None:
    mask = comp["mask"]
    h, w = mask.shape
    if h < 3 or w < 3:
        return None
    if bool(np.any(mask[1:-1, 1:-1])):
        return None
    if not (bool(mask[0, 0]) and bool(mask[0, -1]) and bool(mask[-1, 0]) and bool(mask[-1, -1])):
        return None

    side_arrays = {
        "top": mask[0, 1:-1],
        "bottom": mask[-1, 1:-1],
        "left": mask[1:-1, 0],
        "right": mask[1:-1, -1],
    }
    gap_side = None
    gap_width = None
    for side, arr in side_arrays.items():
        runs = _false_runs(arr)
        if not runs:
            if not bool(arr.all()):
                return None
            continue
        if len(runs) != 1:
            return None
        if gap_side is not None:
            return None
        gap_side = side
        gap_width = int(runs[0][1])
    if gap_side is None or gap_width is None:
        return None

    return {
        "kind": "frame",
        "color": int(comp["color"]),
        "bbox": comp["bbox"],
        "mask": mask,
        "gap_side": gap_side,
        "gap_width": gap_width,
    }


def _full_sides(mask: np.ndarray) -> list[str]:
    sides = []
    if bool(mask[0, :].all()):
        sides.append("top")
    if bool(mask[:, -1].all()):
        sides.append("right")
    if bool(mask[-1, :].all()):
        sides.append("bottom")
    if bool(mask[:, 0].all()):
        sides.append("left")
    return sides


def _sprite_info(comp: dict) -> dict | None:
    mask = comp["mask"]
    sides = _full_sides(mask)
    if len(sides) != 1:
        return None
    side = sides[0]
    h, w = mask.shape
    match_len = w if side in ("top", "bottom") else h
    return {
        "kind": "sprite",
        "color": int(comp["color"]),
        "bbox": comp["bbox"],
        "mask": mask,
        "match_side": side,
        "match_len": int(match_len),
    }


def _rotate_mask_to_side(mask: np.ndarray, target_side: str) -> np.ndarray:
    for k in range(4):
        rot = np.rot90(mask, k)
        if _full_sides(rot) == [target_side]:
            return rot
    raise AssertionError("sprite side could not be rotated to target")


def _placement_in_frame(
    gap_side: str, inner_h: int, inner_w: int, sprite_h: int, sprite_w: int
) -> tuple[int, int]:
    if gap_side == "top":
        return 1 + inner_h - sprite_h, 1 + (inner_w - sprite_w) // 2
    if gap_side == "bottom":
        return 1, 1 + (inner_w - sprite_w) // 2
    if gap_side == "left":
        return 1 + (inner_h - sprite_h) // 2, 1 + inner_w - sprite_w
    if gap_side == "right":
        return 1 + (inner_h - sprite_h) // 2, 1
    raise ValueError(gap_side)


def _render_pair_output(
    frame_mask: np.ndarray,
    frame_color: int,
    sprite_mask: np.ndarray,
    sprite_color: int,
    gap_side: str,
    bg: int,
) -> np.ndarray:
    patch = np.full(frame_mask.shape, bg, dtype=int)
    patch[frame_mask] = int(frame_color)
    inner_h = int(frame_mask.shape[0] - 2)
    inner_w = int(frame_mask.shape[1] - 2)
    sh, sw = sprite_mask.shape
    top, left = _placement_in_frame(gap_side, inner_h, inner_w, int(sh), int(sw))
    for dr, dc in np.argwhere(sprite_mask):
        rr = int(top + dr)
        cc = int(left + dc)
        color = int(sprite_color)
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < patch.shape[0] and 0 <= nc < patch.shape[1] and bool(frame_mask[nr, nc]):
                color = int(frame_color)
                break
        patch[rr, cc] = color
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gapped frame with the uniquely width-matched outside sprite and recolor the wall-touching edge.

    Input structure:
    - The background is the most common color.
    - Several monochrome hollow rectangular frames appear on the grid; each frame has exactly one continuous gap on one side.
    - Separate monochrome filled sprites lie outside the frames; each sprite has exactly one fully filled bbox side, and that side's length matches one frame gap width.

    Rule:
    - Detect every frame, measure its gap width, and record which side contains the gap.
    - Match each frame to the unique outside sprite whose distinguished full bbox side has the same length as the frame gap.
    - Rotate that sprite so its distinguished side faces the gap direction, then place it inside the frame against the opposite inner wall and centered on the perpendicular axis.
    - Recolor any placed sprite pixel that is 4-neighbor adjacent to the frame border with the frame color; keep the other sprite pixels unchanged.
    - Output only the frames and their inserted contents, cropped to the minimal bounding box covering all frames.

    Key insight:
    - The gap width is the pairing key, but the gap direction only becomes usable after noticing that each external sprite has one uniquely identifiable full side whose orientation must be rotated toward the gap.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    frames = []
    sprites = []
    for comp in _components(g, bg):
        frame = _frame_info(comp)
        if frame is not None:
            frames.append(frame)
            continue
        sprite = _sprite_info(comp)
        if sprite is not None:
            sprites.append(sprite)

    assert frames, "expected at least one frame"
    assert len(frames) == len(sprites), "expected one sprite per frame"

    sprites_by_len: dict[int, list[int]] = {}
    for i, sprite in enumerate(sprites):
        sprites_by_len.setdefault(int(sprite["match_len"]), []).append(i)

    min_r = min(int(frame["bbox"][0]) for frame in frames)
    min_c = min(int(frame["bbox"][1]) for frame in frames)
    max_r = max(int(frame["bbox"][2]) for frame in frames)
    max_c = max(int(frame["bbox"][3]) for frame in frames)
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)

    used_sprites: set[int] = set()
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        h, w = int(r1 - r0 + 1), int(c1 - c0 + 1)
        ro, co = int(r0 - min_r), int(c0 - min_c)
        out[ro : ro + h, co : co + w][frame["mask"]] = int(frame["color"])

    for frame in frames:
        gap_width = int(frame["gap_width"])
        candidates = [
            i for i in sprites_by_len.get(gap_width, []) if int(i) not in used_sprites
        ]
        assert len(candidates) == 1, "gap width must identify a unique sprite"
        sprite = sprites[candidates[0]]
        used_sprites.add(int(candidates[0]))

        rot_mask = _rotate_mask_to_side(sprite["mask"], str(frame["gap_side"]))
        fh, fw = frame["mask"].shape
        sh, sw = rot_mask.shape
        inner_h = int(fh - 2)
        inner_w = int(fw - 2)
        assert sh <= inner_h and sw <= inner_w, "sprite must fit inside its frame"

        pair_patch = _render_pair_output(
            frame["mask"],
            int(frame["color"]),
            rot_mask,
            int(sprite["color"]),
            str(frame["gap_side"]),
            bg,
        )
        r0, c0, r1, c1 = map(int, frame["bbox"])
        h, w = int(r1 - r0 + 1), int(c1 - c0 + 1)
        ro, co = int(r0 - min_r), int(c0 - min_c)
        mask = pair_patch != bg
        out[ro : ro + h, co : co + w][mask] = pair_patch[mask]

    return out


def _profile_mask(height: int, width: int, side: str) -> np.ndarray | None:
    if side in ("top", "bottom") and width < 3:
        return None
    if side in ("left", "right") and height < 3:
        return None
    if height < 2 or width < 2:
        return None

    if side in ("top", "bottom"):
        vals = [1] * width
        interior = list(range(1, width - 1))
        full_cols = set(random.sample(interior, random.randint(1, len(interior))))
        for c in range(width):
            if c in (0, width - 1):
                vals[c] = random.randint(1, height - 1)
            elif c in full_cols:
                vals[c] = height
            else:
                vals[c] = random.randint(1, height - 1)
        mask = np.zeros((height, width), dtype=bool)
        for c, depth in enumerate(vals):
            if side == "top":
                mask[:depth, c] = True
            else:
                mask[height - depth :, c] = True
    else:
        vals = [1] * height
        interior = list(range(1, height - 1))
        full_rows = set(random.sample(interior, random.randint(1, len(interior))))
        for r in range(height):
            if r in (0, height - 1):
                vals[r] = random.randint(1, width - 1)
            elif r in full_rows:
                vals[r] = width
            else:
                vals[r] = random.randint(1, width - 1)
        mask = np.zeros((height, width), dtype=bool)
        for r, span in enumerate(vals):
            if side == "left":
                mask[r, :span] = True
            else:
                mask[r, width - span :] = True

    if _full_sides(mask) != [side]:
        return None
    return mask


def _build_frame_mask(
    inner_h: int, inner_w: int, gap_side: str, gap_width: int, gap_offset: int
) -> np.ndarray:
    mask = np.zeros((inner_h + 2, inner_w + 2), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    if gap_side == "top":
        mask[0, 1 + gap_offset : 1 + gap_offset + gap_width] = False
    elif gap_side == "bottom":
        mask[-1, 1 + gap_offset : 1 + gap_offset + gap_width] = False
    elif gap_side == "left":
        mask[1 + gap_offset : 1 + gap_offset + gap_width, 0] = False
    elif gap_side == "right":
        mask[1 + gap_offset : 1 + gap_offset + gap_width, -1] = False
    else:
        raise ValueError(gap_side)
    return mask


def _can_place_bbox(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = occ.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _mark_bbox(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occ[r0 : r0 + h, c0 : c0 + w] = True


def _place_rectangles(boxes: list[tuple[str, int, int]]) -> tuple[int, int, dict[str, tuple[int, int]]] | None:
    total_area = sum((h + 2) * (w + 2) for _name, h, w in boxes)
    max_h = max(h for _name, h, _w in boxes)
    max_w = max(w for _name, _h, w in boxes)
    min_side = max(CANVAS_MIN, max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(total_area)))))

    if min_side > CANVAS_MAX:
        return None

    for _ in range(60):
        H = random.randint(min_side, CANVAS_MAX)
        W = random.randint(min_side, CANVAS_MAX)
        occ = np.zeros((H, W), dtype=bool)
        placements: dict[str, tuple[int, int]] = {}
        ok = True
        for name, h, w in sorted(boxes, key=lambda x: x[1] * x[2], reverse=True):
            candidates = [
                (r, c)
                for r in range(H - h + 1)
                for c in range(W - w + 1)
                if _can_place_bbox(occ, r, c, h, w)
            ]
            if not candidates:
                ok = False
                break
            r0, c0 = random.choice(candidates)
            placements[name] = (int(r0), int(c0))
            _mark_bbox(occ, int(r0), int(c0), int(h), int(w))
        if ok:
            return int(H), int(W), placements
    return None


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    mask = patch != bg
    view = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    view[mask] = patch[mask]


def _paste_mask(grid: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    view = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    view[mask] = int(color)


def _sprite_spec(gap_width: int, other_dim: int, match_side: str) -> dict | None:
    if match_side in ("top", "bottom"):
        height, width = other_dim, gap_width
    else:
        height, width = gap_width, other_dim
    mask = _profile_mask(height, width, match_side)
    if mask is None:
        return None
    return {
        "input_mask": mask,
        "match_side": match_side,
        "gap_width": int(gap_width),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, 9)
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    gap_widths = random.sample(GAP_WIDTH_POOL, pair_count)
    other_choices = [x for x in OTHER_DIM_POOL if x not in gap_widths]
    if not other_choices:
        return None

    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)
    if len(colors) < 2 * pair_count:
        return None

    specs = []
    for i, gap_width in enumerate(gap_widths):
        frame_color = int(colors[2 * i])
        sprite_color = int(colors[2 * i + 1])
        gap_side = random.choice(SIDES)
        match_side = random.choice(SIDES)
        other_dim = random.choice(other_choices)

        sprite = _sprite_spec(int(gap_width), int(other_dim), match_side)
        if sprite is None:
            return None

        output_mask = _rotate_mask_to_side(sprite["input_mask"], gap_side)
        sh, sw = output_mask.shape

        side_margin = random.randint(1, 2)
        gap_space = random.randint(1, 3)
        if gap_side in ("top", "bottom"):
            inner_h = int(sh + gap_space)
            inner_w = int(sw + 2 * side_margin)
            gap_offset = random.randint(0, inner_w - gap_width)
        else:
            inner_h = int(sh + 2 * side_margin)
            inner_w = int(sw + gap_space)
            gap_offset = random.randint(0, inner_h - gap_width)

        frame_mask = _build_frame_mask(inner_h, inner_w, gap_side, int(gap_width), gap_offset)
        output_patch = _render_pair_output(
            frame_mask, frame_color, output_mask, sprite_color, gap_side, bg
        )

        specs.append(
            {
                "name": f"pair{i}",
                "frame_color": frame_color,
                "sprite_color": sprite_color,
                "gap_side": gap_side,
                "gap_width": int(gap_width),
                "frame_mask": frame_mask,
                "sprite_input_mask": sprite["input_mask"],
                "output_patch": output_patch,
            }
        )

    boxes = []
    for spec in specs:
        fh, fw = spec["frame_mask"].shape
        sh, sw = spec["sprite_input_mask"].shape
        boxes.append((f"{spec['name']}_frame", int(fh), int(fw)))
        boxes.append((f"{spec['name']}_sprite", int(sh), int(sw)))

    placement = _place_rectangles(boxes)
    if placement is None:
        return None
    H, W, placements = placement

    grid = np.full((H, W), bg, dtype=int)
    frame_positions = []
    for spec in specs:
        fr, fc = placements[f"{spec['name']}_frame"]
        sr, sc = placements[f"{spec['name']}_sprite"]
        frame_positions.append((fr, fc, spec["frame_mask"].shape[0], spec["frame_mask"].shape[1]))
        _paste_mask(grid, spec["frame_mask"], int(spec["frame_color"]), int(fr), int(fc))
        _paste_mask(
            grid, spec["sprite_input_mask"], int(spec["sprite_color"]), int(sr), int(sc)
        )

    vals, counts = np.unique(grid, return_counts=True)
    if int(vals[int(np.argmax(counts))]) != bg:
        return None

    min_r = min(fr for fr, _fc, _h, _w in frame_positions)
    min_c = min(fc for _fr, fc, _h, _w in frame_positions)
    max_r = max(fr + h - 1 for fr, _fc, h, _w in frame_positions)
    max_c = max(fc + w - 1 for _fr, fc, _h, w in frame_positions)
    expected = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)

    for spec in specs:
        fr, fc = placements[f"{spec['name']}_frame"]
        patch = spec["output_patch"]
        _paste_patch(expected, patch, int(fr - min_r), int(fc - min_c), bg)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid sample")
