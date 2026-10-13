import random
from typing import Iterable

import numpy as np


BG = 0
FRAME = 5
SPRITE_COLORS = (1, 2, 3, 4)


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for idx, coords in enumerate(_components_mask(grid == FRAME)):
        bbox = _bbox_from_coords(coords)
        patch = _crop_bbox(grid, bbox)
        ph, pw = patch.shape
        if ph < 3 or pw < 3:
            continue
        border = np.ones((ph, pw), dtype=bool)
        border[1:-1, 1:-1] = False
        if not np.all(patch[border] == FRAME):
            continue
        if not np.all(patch[~border] == BG):
            continue
        if int(coords.shape[0]) != int(border.sum()):
            continue
        r0, c0, r1, c1 = bbox
        border_coords = [(r0 + rr, c0 + cc) for rr, cc in np.argwhere(border)]
        frames.append(
            {
                "id": idx,
                "bbox": bbox,
                "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                "inner_h": ph - 2,
                "inner_w": pw - 2,
                "border_coords": border_coords,
            }
        )
    frames.sort(key=lambda f: (f["inner_bbox"][0], f["inner_bbox"][1]))
    for new_id, frame in enumerate(frames):
        frame["id"] = new_id
    return frames


def _find_sprites(grid: np.ndarray) -> list[dict]:
    sprites = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {BG, FRAME}):
        for coords in _components_mask(grid == color):
            bbox = _bbox_from_coords(coords)
            r0, c0, r1, c1 = bbox
            rel = coords - np.array([[r0, c0]])
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            mask = np.zeros((h, w), dtype=bool)
            mask[rel[:, 0], rel[:, 1]] = True
            sprites.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox,
                    "rel_coords": rel,
                    "mask": mask,
                    "h": h,
                    "w": w,
                }
            )
    sprites.sort(key=lambda s: (s["bbox"][0], s["bbox"][1], s["color"]))
    return sprites


def _eligible_frames(sprite: dict, frames: list[dict]) -> list[dict]:
    r0, c0, r1, c1 = sprite["bbox"]
    eligible = []
    for frame in frames:
        ir0, ic0, _ir1, ic1 = frame["inner_bbox"]
        if r1 >= ir0:
            continue
        if sprite["h"] > frame["inner_h"] or sprite["w"] > frame["inner_w"]:
            continue
        if c0 < ic0 or c1 > ic1:
            continue
        eligible.append(frame)
    eligible.sort(key=lambda f: (f["inner_bbox"][0], f["inner_bbox"][1]))
    return eligible


def _render_from_plan(shape: tuple[int, int], frames: list[dict], sprites: list[dict], plan: dict[int, dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for frame in frames:
        border_color = int(plan.get(("frame", frame["id"]), FRAME))
        for r, c in frame["border_coords"]:
            out[r, c] = border_color
    for sprite in sprites:
        target_r, target_c = plan[sprite["color"]]["target"]
        for rr, cc in sprite["rel_coords"]:
            ar = target_r + int(rr)
            ac = target_c + int(cc)
            if out[ar, ac] != BG:
                raise AssertionError("sprite placement overlap")
            out[ar, ac] = int(sprite["color"])
    return out


def _solve_layout(grid: np.ndarray) -> tuple[list[dict], list[dict], dict]:
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    sprites = _find_sprites(g)
    occupied_frames: set[int] = set()
    plan: dict = {}
    h, _w = g.shape
    for sprite in sprites:
        candidates = [frame for frame in _eligible_frames(sprite, frames) if frame["id"] not in occupied_frames]
        if candidates:
            frame = candidates[0]
            ir0, ic0, ir1, _ic1 = frame["inner_bbox"]
            target_r = ir1 - sprite["h"] + 1
            target_c = ic0 + (frame["inner_w"] - sprite["w"]) // 2
            occupied_frames.add(frame["id"])
            plan[sprite["color"]] = {"target": (target_r, target_c), "frame_id": frame["id"]}
            plan[("frame", frame["id"])] = int(sprite["color"])
        else:
            target_r = h - sprite["h"]
            target_c = sprite["bbox"][1]
            plan[sprite["color"]] = {"target": (target_r, target_c), "frame_id": None}
    for frame in frames:
        plan.setdefault(("frame", frame["id"]), FRAME)
    return frames, sprites, plan


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let falling monochrome sprites be captured by size-compatible hollow frames and recolor those frames.

    Input structure:
    - The background is black.
    - Several gray one-cell-thick hollow rectangular frames sit in the lower half; their interior sizes differ.
    - Several small irregular solid monochrome sprites of distinct non-gray colors sit above the frames.

    Rule:
    - Extract every sprite as a rigid 4-connected object and sort sprites by top row, then by left column.
    - For each sprite, look downward for the highest still-empty frame whose interior fully contains the sprite's current column span and whose interior width/height can contain the sprite's bounding box.
    - A captured sprite is moved into that frame, bottom-aligned to the frame interior and horizontally centered, using left bias when exact centering is impossible; if no empty eligible frame remains, the sprite falls to the grid bottom while keeping its column position.
    - Every frame that captures a sprite has its entire border recolored to that sprite's color; untouched frames stay gray.

    Key insight:
    - Capture depends on the sprite's bounding-box size and original column span, while processing order decides which earlier sprite can occupy a frame before a later sprite reaches it.
    """
    g = np.asarray(grid, dtype=int)
    frames, sprites, plan = _solve_layout(g)
    return _render_from_plan(g.shape, frames, sprites, plan)


def _split_extra(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1)) if parts > 1 else []
    values = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _random_shape(h: int, w: int) -> np.ndarray:
    steps = ["D"] * (h - 1) + ["R"] * (w - 1)
    random.shuffle(steps)
    r = c = 0
    cells = {(0, 0)}
    for step in steps:
        if step == "D":
            r += 1
        else:
            c += 1
        cells.add((r, c))
    target = len(cells)
    max_area = min(8, h * w)
    if max_area > target:
        upper = max_area if max_area == h * w == target else max(target + 1, max_area)
        target = random.randint(target, max_area if max_area < h * w else max_area - (1 if h * w > len(cells) else 0))
        target = max(target, len(cells))
    frontier = set()
    while len(cells) < target:
        frontier.clear()
        for rr, cc in cells:
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(sorted(frontier)))
    mask = np.zeros((h, w), dtype=bool)
    for rr, cc in cells:
        mask[rr, cc] = True
    assert mask[0, 0] and mask[h - 1, w - 1]
    assert 3 <= int(mask.sum()) <= 8
    return mask


def _dims_that_fit(max_h: int, max_w: int, max_area_hard: int = 8) -> list[tuple[int, int]]:
    dims = []
    for h in range(2, max_h + 1):
        for w in range(2, max_w + 1):
            min_area = h + w - 1
            if min_area > max_area_hard:
                continue
            if h * w < 3:
                continue
            dims.append((h, w))
    random.shuffle(dims)
    return dims


def _frame_columns_for_box(frame: dict, width: int) -> list[int]:
    ic0 = frame["inner_bbox"][1]
    ic1 = frame["inner_bbox"][3]
    return list(range(ic0, ic1 - width + 2))


def _full_bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _choose_targeted_sprite(frame: dict, frames: list[dict], require_bottom_safe: bool) -> tuple[np.ndarray, int, int, int]:
    room_below = 10 ** 9 if not require_bottom_safe else max(1, _GEN_H - frame["bbox"][2] - 1)
    for h, w in _dims_that_fit(min(frame["inner_h"], room_below), frame["inner_w"]):
        valid_cols = []
        for c0 in _frame_columns_for_box(frame, w):
            probe = {"bbox": (0, c0, h - 1, c0 + w - 1), "h": h, "w": w}
            eligible = [f["id"] for f in _eligible_frames(probe, frames)]
            if eligible != [frame["id"]]:
                continue
            if require_bottom_safe:
                bottom_box = (_GEN_H - h, c0, _GEN_H - 1, c0 + w - 1)
                if any(_full_bbox_overlap(bottom_box, other["bbox"]) for other in frames):
                    continue
            valid_cols.append(c0)
        if valid_cols:
            c0 = random.choice(valid_cols)
            return _random_shape(h, w), h, w, c0
    raise RuntimeError("no targeted sprite fits")


def _choose_stray_sprite(frames: list[dict]) -> tuple[np.ndarray, int, int, int]:
    global _GEN_W
    for h, w in _dims_that_fit(4, min(5, _GEN_W - 2)):
        valid_cols = []
        for c0 in range(1, _GEN_W - w):
            probe = {"bbox": (0, c0, h - 1, c0 + w - 1), "h": h, "w": w}
            if _eligible_frames(probe, frames):
                continue
            bottom_box = (_GEN_H - h, c0, _GEN_H - 1, c0 + w - 1)
            if any(_full_bbox_overlap(bottom_box, frame["bbox"]) for frame in frames):
                continue
            valid_cols.append(c0)
        if valid_cols:
            c0 = random.choice(valid_cols)
            return _random_shape(h, w), h, w, c0
    raise RuntimeError("no stray sprite fits")


def _paint_shape(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    for rr, cc in np.argwhere(mask):
        grid[top + int(rr), left + int(cc)] = int(color)


def _make_frame_spec(inner_h: int, inner_w: int, top: int, left: int, frame_id: int) -> dict:
    return {
        "id": frame_id,
        "bbox": (top, left, top + inner_h + 1, left + inner_w + 1),
        "inner_bbox": (top + 1, left + 1, top + inner_h, left + inner_w),
        "inner_h": inner_h,
        "inner_w": inner_w,
        "border_coords": (
            [(top, c) for c in range(left, left + inner_w + 2)]
            + [(top + inner_h + 1, c) for c in range(left, left + inner_w + 2)]
            + [(r, left) for r in range(top + 1, top + inner_h + 1)]
            + [(r, left + inner_w + 1) for r in range(top + 1, top + inner_h + 1)]
        ),
    }


def _build_frames(n_frames: int) -> tuple[list[dict], int, int, int]:
    large_sizes = [(5, 4), (4, 5), (5, 5)]
    other_sizes = [(3, 3), (3, 4), (4, 3), (4, 4), (5, 3), (3, 5)]
    comp_size = random.choice(large_sizes)
    other = [random.choice(other_sizes) for _ in range(n_frames - 1)]
    sizes = [comp_size] + other
    random.shuffle(sizes)
    comp_index = sizes.index(comp_size)
    full_ws = [w + 2 for _, w in sizes]
    base_gap = 2
    base_margin = 1
    min_w = sum(full_ws) + base_gap * (n_frames - 1) + 2 * base_margin
    if min_w > 28:
        raise RuntimeError("layout too wide")
    w = random.randint(max(18, min_w), min(28, min_w + 6))
    slack = w - min_w
    gaps = [base_margin] + [base_gap] * (n_frames - 1) + [base_margin]
    extras = _split_extra(slack, len(gaps))
    gaps = [g + e for g, e in zip(gaps, extras)]

    h = random.randint(20, 26)
    frames = []
    col = gaps[0]
    for idx, (ih, iw) in enumerate(sizes):
        full_h = ih + 2
        low = h // 2
        high = h - full_h - 1
        if idx == comp_index:
            high = min(high, h - full_h - 4)
        if high < low:
            raise RuntimeError("not enough vertical room for frame")
        top = random.randint(low, high)
        frames.append(_make_frame_spec(ih, iw, top, col, idx))
        col += iw + 2 + gaps[idx + 1]
    frames.sort(key=lambda f: f["bbox"][1])
    for idx, frame in enumerate(frames):
        frame["id"] = idx
    comp_frame = min(frames, key=lambda f: (-(f["inner_h"] * f["inner_w"]), f["bbox"][1]))
    return frames, comp_frame["id"], h, w


def _assign_sprite_rows(sprites: list[dict], h: int) -> None:
    ordered = [s for s in sprites if s["role"] == "comp_win"]
    middle = [s for s in sprites if s["role"] not in {"comp_win", "comp_lose"}]
    random.shuffle(middle)
    ordered.extend(middle)
    ordered.extend([s for s in sprites if s["role"] == "comp_lose"])
    total_heights = sum(s["h"] for s in ordered)
    top_band = h // 2 - 2
    min_needed = 1 + total_heights + (len(ordered) - 1) * 1
    if min_needed > top_band:
        raise RuntimeError("not enough room for sprites")
    slack = top_band - min_needed
    gaps = [1] + [1] * (len(ordered) - 1)
    extras = _split_extra(slack, len(gaps))
    gaps = [g + e for g, e in zip(gaps, extras)]
    row = gaps[0]
    for idx, sprite in enumerate(ordered):
        sprite["top"] = row
        row += sprite["h"]
        if idx + 1 < len(ordered):
            row += gaps[idx + 1]


def _sprite_from_mask(mask: np.ndarray, color: int, top: int, left: int) -> dict:
    h, w = mask.shape
    coords = np.argwhere(mask) + np.array([[top, left]])
    return {
        "color": int(color),
        "bbox": (top, left, top + h - 1, left + w - 1),
        "coords": coords,
        "rel_coords": np.argwhere(mask),
        "mask": mask,
        "h": h,
        "w": w,
    }


def _render_expected(shape: tuple[int, int], frames: list[dict], sprites: list[dict], roles: dict[int, dict]) -> np.ndarray:
    plan = {}
    for frame in frames:
        plan[("frame", frame["id"])] = roles.get(("frame", frame["id"]), FRAME)
    for sprite in sprites:
        if roles[sprite["color"]]["frame_id"] is None:
            plan[sprite["color"]] = {"target": (shape[0] - sprite["h"], sprite["bbox"][1]), "frame_id": None}
        else:
            frame = next(f for f in frames if f["id"] == roles[sprite["color"]]["frame_id"])
            ir0, ic0, ir1, _ic1 = frame["inner_bbox"]
            target = (ir1 - sprite["h"] + 1, ic0 + (frame["inner_w"] - sprite["w"]) // 2)
            plan[sprite["color"]] = {"target": target, "frame_id": frame["id"]}
    return _render_from_plan(shape, frames, sprites, plan)


def _validate_semantics(grid: np.ndarray, frames: list[dict], sprites: list[dict], roles: dict) -> None:
    out = main(grid)
    expected = _render_expected(grid.shape, frames, sprites, roles)
    assert np.array_equal(out, expected)
    assert 3 <= len(frames) <= 4
    assert 3 <= len(sprites) <= 4
    assert any(roles[s["color"]]["frame_id"] is None for s in sprites)
    assert any(roles[s["color"]]["frame_id"] is not None for s in sprites)
    comp_win = next(s for s in sprites if roles[s["color"]]["role"] == "comp_win")
    comp_lose = next(s for s in sprites if roles[s["color"]]["role"] == "comp_lose")
    assert comp_win["bbox"][0] < comp_lose["bbox"][0]
    lose_eligible = [f["id"] for f in _eligible_frames(comp_lose, frames)]
    win_eligible = [f["id"] for f in _eligible_frames(comp_win, frames)]
    assert lose_eligible == [roles[comp_lose["color"]]["blocked_by"]]
    assert win_eligible == [roles[comp_win["color"]]["frame_id"]]
    for sprite in sprites:
        role = roles[sprite["color"]]["role"]
        eligible = [f["id"] for f in _eligible_frames(sprite, frames)]
        if role == "stray":
            assert eligible == []
        elif role == "comp_lose":
            assert roles[sprite["color"]]["frame_id"] is None
        else:
            assert eligible == [roles[sprite["color"]]["frame_id"]]
    for frame in frames:
        expected_color = roles.get(("frame", frame["id"]), FRAME)
        for r, c in frame["border_coords"]:
            assert int(out[r, c]) == int(expected_color)


_GEN_H = 0
_GEN_W = 0


def generate_input() -> np.ndarray:
    for _ in range(300):
        try:
            return _generate_attempt()
        except Exception:
            continue
    raise RuntimeError("failed to generate valid puzzle input")


def _generate_attempt() -> np.ndarray:
    global _GEN_H, _GEN_W
    scenario = random.choices(
        [
            {"n_frames": 3, "roles": ["comp_win", "comp_lose", "capture"]},
            {"n_frames": 3, "roles": ["comp_win", "comp_lose", "capture", "stray"]},
            {"n_frames": 4, "roles": ["comp_win", "comp_lose", "capture", "capture"]},
        ],
        weights=[2, 4, 4],
        k=1,
    )[0]

    frames, comp_frame_id, _GEN_H, _GEN_W = _build_frames(scenario["n_frames"])
    comp_frame = next(f for f in frames if f["id"] == comp_frame_id)
    free_frames = [f for f in frames if f["id"] != comp_frame_id]
    if len(free_frames) < scenario["roles"].count("capture"):
        raise RuntimeError("not enough distinct frames for capture roles")

    colors = list(SPRITE_COLORS)
    random.shuffle(colors)
    colors = colors[: len(scenario["roles"])]

    sprite_specs = []
    roles: dict = {("frame", frame["id"]): FRAME for frame in frames}

    capture_frames = free_frames[: scenario["roles"].count("capture")]
    capture_iter = iter(capture_frames)
    for color, role in zip(colors, scenario["roles"]):
        if role == "comp_win":
            mask, hh, ww, left = _choose_targeted_sprite(comp_frame, frames, require_bottom_safe=False)
            sprite_specs.append({"color": color, "role": role, "mask": mask, "h": hh, "w": ww, "left": left, "frame_id": comp_frame_id})
            roles[color] = {"role": role, "frame_id": comp_frame_id}
            roles[("frame", comp_frame_id)] = color
        elif role == "comp_lose":
            mask, hh, ww, left = _choose_targeted_sprite(comp_frame, frames, require_bottom_safe=True)
            sprite_specs.append({"color": color, "role": role, "mask": mask, "h": hh, "w": ww, "left": left, "frame_id": None, "blocked_by": comp_frame_id})
            roles[color] = {"role": role, "frame_id": None, "blocked_by": comp_frame_id}
        elif role == "capture":
            frame = next(capture_iter)
            mask, hh, ww, left = _choose_targeted_sprite(frame, frames, require_bottom_safe=False)
            sprite_specs.append({"color": color, "role": role, "mask": mask, "h": hh, "w": ww, "left": left, "frame_id": frame["id"]})
            roles[color] = {"role": role, "frame_id": frame["id"]}
            roles[("frame", frame["id"])] = color
        else:
            mask, hh, ww, left = _choose_stray_sprite(frames)
            sprite_specs.append({"color": color, "role": role, "mask": mask, "h": hh, "w": ww, "left": left, "frame_id": None})
            roles[color] = {"role": role, "frame_id": None}

    _assign_sprite_rows(sprite_specs, _GEN_H)
    grid = np.zeros((_GEN_H, _GEN_W), dtype=int)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        grid[r0:r1 + 1, c0] = FRAME
        grid[r0:r1 + 1, c1] = FRAME
        grid[r0, c0:c1 + 1] = FRAME
        grid[r1, c0:c1 + 1] = FRAME

    sprites = []
    for spec in sprite_specs:
        _paint_shape(grid, spec["mask"], spec["top"], spec["left"], spec["color"])
        sprites.append(_sprite_from_mask(spec["mask"], spec["color"], spec["top"], spec["left"]))

    _validate_semantics(grid, frames, sprites, roles)
    return grid
