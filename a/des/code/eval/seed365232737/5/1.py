import random
from collections import deque

import numpy as np


_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
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
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
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


def _is_frame_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    mask = _tight_mask(coords)
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _rects_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + padding < br0
        or br1 + padding < ar0
        or ac1 + padding < bc0
        or bc1 + padding < ac0
    )


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _transform_mask(mask: np.ndarray, depth: int) -> np.ndarray:
    if depth == 1:
        return mask.copy()
    if depth == 2:
        return np.fliplr(mask)
    if depth == 3:
        return np.rot90(mask, 2)
    raise ValueError(f"unsupported depth: {depth}")


def _paint_mask(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    sub[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    frames: list[dict] = []
    nonframes: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        frame_comp = None
        for coords in _component_coords(g == color):
            if _is_frame_component(coords):
                if frame_comp is None:
                    frame_comp = coords
                else:
                    raise AssertionError("each color should define at most one frame")
            else:
                nonframes.append({"color": int(color), "coords": coords})
        if frame_comp is not None:
            r0, c0, r1, c1 = _bbox(frame_comp)
            frames.append({"color": int(color), "bbox": (r0, c0, r1, c1)})

    assert frames, "expected at least one rectangular frame"
    frame_by_color = {frame["color"]: frame for frame in frames}

    sprites: dict[int, dict] = {}
    frame_boxes = [frame["bbox"] for frame in frames]
    for comp in nonframes:
        color = int(comp["color"])
        if color not in frame_by_color:
            continue
        box = _bbox(comp["coords"])
        if any(_rects_intersect(box, frame_box, padding=0) for frame_box in frame_boxes):
            continue
        if color not in sprites or len(comp["coords"]) > len(sprites[color]["coords"]):
            sprites[color] = {"coords": comp["coords"], "mask": _tight_mask(comp["coords"])}

    for frame in frames:
        frame["depth"] = 1 + sum(
            1 for other in frames if other is not frame and _contains(other["bbox"], frame["bbox"])
        )

    out = g.copy()
    for sprite in sprites.values():
        out[tuple(sprite["coords"].T)] = bg

    for frame in frames:
        sprite = sprites.get(int(frame["color"]))
        if sprite is None:
            continue
        mask = _transform_mask(sprite["mask"], int(frame["depth"]))
        r0, c0, r1, c1 = frame["bbox"]
        ih = r1 - r0 - 1
        iw = c1 - c0 - 1
        mh, mw = mask.shape
        assert mh <= ih and mw <= iw, "sprite must fit inside its frame"
        _paint_mask(out, mask, int(frame["color"]), r0 + 1, c0 + 1)
    return out


def _random_sprite_mask(require_hflip_change: bool = False, require_rot_change: bool = False) -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(2, 3)
        w = random.randint(2, 3)
        area = random.randint(3, min(6, h * w))
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = {nb for nb in _neighbors_in_bounds(start, h, w)}
        while frontier and len(cells) < area:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nb in _neighbors_in_bounds(cell, h, w):
                if nb not in cells:
                    frontier.add(nb)
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        pts = np.argwhere(mask)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if require_hflip_change and np.array_equal(mask, np.fliplr(mask)):
            continue
        if require_rot_change and np.array_equal(mask, np.rot90(mask, 2)):
            continue
        return mask
    return None


def _neighbors_in_bounds(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in _NEIGHBORS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _build_chain(colors: list[int]) -> list[dict] | None:
    sprite1 = _random_sprite_mask(require_hflip_change=True, require_rot_change=True)
    sprite2 = _random_sprite_mask(require_hflip_change=True, require_rot_change=False)
    sprite3 = _random_sprite_mask(require_hflip_change=False, require_rot_change=True)
    if sprite1 is None or sprite2 is None or sprite3 is None:
        return None

    s1h, s1w = sprite1.shape
    s2h, s2w = sprite2.shape
    s3h, s3w = sprite3.shape

    inner_in_h = s3h + random.randint(1, 2)
    inner_in_w = s3w + random.randint(1, 2)
    inner_h = inner_in_h + 2
    inner_w = inner_in_w + 2

    off2_r = s2h + 1
    off2_c = s2w + 1
    middle_in_h = off2_r + inner_h + random.randint(1, 2)
    middle_in_w = off2_c + inner_w + random.randint(1, 2)
    middle_h = middle_in_h + 2
    middle_w = middle_in_w + 2

    off1_r = s1h + 1
    off1_c = s1w + 1
    outer_in_h = off1_r + middle_h + random.randint(1, 2)
    outer_in_w = off1_c + middle_w + random.randint(1, 2)
    outer_h = outer_in_h + 2
    outer_w = outer_in_w + 2

    if outer_h > 20 or outer_w > 20:
        return None

    outer = {"color": int(colors[0]), "depth": 1, "bbox": (0, 0, outer_h - 1, outer_w - 1), "sprite": sprite1}
    mr0 = 1 + off1_r
    mc0 = 1 + off1_c
    middle = {
        "color": int(colors[1]),
        "depth": 2,
        "bbox": (mr0, mc0, mr0 + middle_h - 1, mc0 + middle_w - 1),
        "sprite": sprite2,
    }
    ir0 = mr0 + 1 + off2_r
    ic0 = mc0 + 1 + off2_c
    inner = {
        "color": int(colors[2]),
        "depth": 3,
        "bbox": (ir0, ic0, ir0 + inner_h - 1, ic0 + inner_w - 1),
        "sprite": sprite3,
    }
    return [outer, middle, inner]


def _build_standalone_frame(color: int, matched: bool) -> dict | None:
    sprite = None
    if matched:
        sprite = _random_sprite_mask(require_hflip_change=True, require_rot_change=True)
        if sprite is None:
            return None
        sh, sw = sprite.shape
        inner_h = sh + random.randint(1, 3)
        inner_w = sw + random.randint(1, 3)
    else:
        inner_h = random.randint(4, 7)
        inner_w = random.randint(4, 7)
    return {
        "color": int(color),
        "depth": 1,
        "bbox": (0, 0, inner_h + 1, inner_w + 1),
        "sprite": sprite,
    }


def _shift_frame(frame: dict, dr: int, dc: int) -> dict:
    r0, c0, r1, c1 = frame["bbox"]
    return {
        "color": int(frame["color"]),
        "depth": int(frame["depth"]),
        "bbox": (r0 + dr, c0 + dc, r1 + dr, c1 + dc),
        "sprite": None if frame["sprite"] is None else frame["sprite"].copy(),
    }


def _layout_groups(chain_frames: list[dict], extras: list[dict]) -> list[dict]:
    chain_box = chain_frames[0]["bbox"]
    chain_r0, chain_c0, chain_r1, chain_c1 = chain_box
    chain_h = chain_r1 + 1
    chain_w = chain_c1 + 1
    placements: list[tuple[dict, int, int]] = [(frame, 0, 0) for frame in chain_frames]
    top_level_boxes = [(chain_r0, chain_c0, chain_r1, chain_c1)]

    sides = ["right", "left", "below", "above"]
    random.shuffle(sides)
    for frame, side in zip(extras, sides):
        r0, c0, r1, c1 = frame["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        gap = random.randint(2, 5)
        if side == "right":
            dr = random.randint(-1, max(1, chain_h - h + 1))
            dc = chain_w + gap
        elif side == "left":
            dr = random.randint(-1, max(1, chain_h - h + 1))
            dc = -w - gap
        elif side == "below":
            dr = chain_h + gap
            dc = random.randint(-1, max(1, chain_w - w + 1))
        else:
            dr = -h - gap
            dc = random.randint(-1, max(1, chain_w - w + 1))
        top_level_boxes.append((dr, dc, dr + h - 1, dc + w - 1))
        placements.append((frame, dr, dc))

    min_r = min(frame["bbox"][0] + dr for frame, dr, _ in placements)
    min_c = min(frame["bbox"][1] + dc for frame, _, dc in placements)
    shifted = [_shift_frame(frame, dr - min_r, dc - min_c) for frame, dr, dc in placements]

    normalized_boxes = [
        (r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c) for r0, c0, r1, c1 in top_level_boxes
    ]
    for i in range(len(normalized_boxes)):
        for j in range(i + 1, len(normalized_boxes)):
            if _rects_intersect(normalized_boxes[i], normalized_boxes[j], padding=1):
                raise AssertionError("layout should keep top-level frames separated")
    return shifted


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    color = int(frame["color"])
    r0, c0, r1, c1 = frame["bbox"]
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _scene_bounds(frames: list[dict]) -> tuple[int, int]:
    h = max(frame["bbox"][2] for frame in frames) + 1
    w = max(frame["bbox"][3] for frame in frames) + 1
    return int(h), int(w)


def _shift_all(frames: list[dict], dr: int, dc: int) -> list[dict]:
    return [_shift_frame(frame, dr, dc) for frame in frames]


def _find_sprite_spot(
    grid: np.ndarray,
    sprite: np.ndarray,
    blocked_boxes: list[tuple[int, int, int, int]],
    bg: int,
) -> tuple[int, int] | None:
    h, w = grid.shape
    sh, sw = sprite.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h - sh + 1):
        for c in range(w - sw + 1):
            box = (r, c, r + sh - 1, c + sw - 1)
            if any(_rects_intersect(box, other, padding=1) for other in blocked_boxes):
                continue
            sub = grid[r : r + sh, c : c + sw]
            if np.any(sprite & (sub != bg)):
                continue
            candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _build_expected_output(base: np.ndarray, frames: list[dict], bg: int) -> np.ndarray:
    out = base.copy()
    for frame in frames:
        if frame["sprite"] is None:
            continue
        mask = _transform_mask(frame["sprite"], int(frame["depth"]))
        r0, c0, r1, c1 = frame["bbox"]
        ih = r1 - r0 - 1
        iw = c1 - c0 - 1
        if mask.shape[0] > ih or mask.shape[1] > iw:
            raise AssertionError("generated sprite does not fit target frame")
        _paint_mask(out, mask, int(frame["color"]), r0 + 1, c0 + 1)
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(10)
    extra_count = 1 + int(random.random() < 0.5)
    n_frames = 3 + extra_count
    colors = random.sample([c for c in range(10) if c != bg], n_frames)

    chain = _build_chain(colors[:3])
    if chain is None:
        return None

    extras: list[dict] = []
    unmatched = _build_standalone_frame(colors[3], matched=False)
    if unmatched is None:
        return None
    extras.append(unmatched)
    if extra_count == 2:
        matched = _build_standalone_frame(colors[4], matched=bool(random.getrandbits(1)))
        if matched is None:
            return None
        extras.append(matched)

    try:
        frames = _layout_groups(chain, extras)
    except AssertionError:
        return None

    scene_h, scene_w = _scene_bounds(frames)
    pad_top = random.randint(3, 7)
    pad_bottom = random.randint(3, 7)
    pad_left = random.randint(3, 7)
    pad_right = random.randint(3, 7)
    frames = _shift_all(frames, pad_top, pad_left)
    h = scene_h + pad_top + pad_bottom
    w = scene_w + pad_left + pad_right

    base = np.full((h, w), bg, dtype=int)
    for frame in frames:
        _draw_frame(base, frame)

    inp = base.copy()
    blocked_boxes = [frame["bbox"] for frame in frames]
    sprite_frames = [frame for frame in frames if frame["sprite"] is not None]
    sprite_frames.sort(key=lambda frame: int(frame["sprite"].size), reverse=True)
    for frame in sprite_frames:
        pos = _find_sprite_spot(inp, frame["sprite"], blocked_boxes, bg)
        if pos is None:
            return None
        sr, sc = pos
        _paint_mask(inp, frame["sprite"], int(frame["color"]), sr, sc)
        blocked_boxes.append((sr, sc, sr + frame["sprite"].shape[0] - 1, sc + frame["sprite"].shape[1] - 1))

    expected = _build_expected_output(base, frames, bg)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if _get_bg(inp) != bg:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
