from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
SPRITE_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    arr = arr - np.array([r0, c0], dtype=int)
    return tuple(sorted((int(r), int(c)) for r, c in arr))


def _rot90(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h = max(r for r, _ in shape) + 1
    return _normalize([(c, h - 1 - r) for r, c in shape])


def _flip_h(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    w = max(c for _, c in shape) + 1
    return _normalize([(r, w - 1 - c) for r, c in shape])


def _d4(shape: tuple[tuple[int, int], ...]) -> set[tuple[tuple[int, int], ...]]:
    out = set()
    cur = shape
    for _ in range(4):
        out.add(cur)
        cur = _rot90(cur)
    cur = _flip_h(shape)
    for _ in range(4):
        out.add(cur)
        cur = _rot90(cur)
    return out


def _sprite_info(grid: np.ndarray) -> list[dict]:
    infos = []
    for color in sorted(set(map(int, grid.ravel())) - {BG, BLUE}):
        comps = _components(grid == color)
        for comp in comps:
            shape = _normalize(comp)
            infos.append(
                {
                    "color": int(color),
                    "coords": comp,
                    "shape": shape,
                    "variants": _d4(shape),
                }
            )
    return infos


def _object_info(grid: np.ndarray) -> list[dict]:
    infos = []
    for comp in _components(grid == BLUE):
        r0, c0, r1, c1 = _bbox(comp)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        missing = np.argwhere(patch != BLUE)
        assert len(missing) > 0, "each blue object must have a boundary gap"
        assert np.all(patch[patch != BLUE] == BG), "object bbox may only contain blue or background"

        side_hits = []
        if np.all(missing[:, 0] == 0):
            side_hits.append("top")
        if np.all(missing[:, 0] == h - 1):
            side_hits.append("bottom")
        if np.all(missing[:, 1] == 0):
            side_hits.append("left")
        if np.all(missing[:, 1] == w - 1):
            side_hits.append("right")
        assert len(side_hits) == 1, "gap must lie on exactly one bbox side"
        side = side_hits[0]

        miss_shape = _normalize(missing)
        infos.append(
            {
                "bbox": (r0, c0, r1, c1),
                "side": side,
                "gap_shape": miss_shape,
                "h": h,
                "w": w,
            }
        )
    return infos


def _target_position(h: int, w: int, side: str, grid_shape: tuple[int, int]) -> tuple[int, int]:
    gh, gw = grid_shape
    if side == "top":
        return gh - h, 0
    if side == "bottom":
        return 0, 0
    if side == "left":
        return 0, gw - w
    return 0, 0


def _moved_origin(obj: dict, grid_shape: tuple[int, int]) -> tuple[int, int]:
    gh, gw = grid_shape
    r0, c0, _r1, _c1 = obj["bbox"]
    h = obj["h"]
    w = obj["w"]
    if obj["side"] == "top":
        return gh - h, c0
    if obj["side"] == "bottom":
        return 0, c0
    if obj["side"] == "left":
        return r0, gw - w
    return r0, 0


def _render_object_patch(h: int, w: int, side: str, color: int) -> np.ndarray:
    patch = np.full((h, w), BLUE, dtype=int)
    if side == "top":
        patch[0, :] = color
    elif side == "bottom":
        patch[-1, :] = color
    elif side == "left":
        patch[:, 0] = color
    else:
        patch[:, -1] = color
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each blue rectangle's boundary gap to an external monochrome sprite.
    Recolor the full gap-side edge with the sprite color, then slide the object
    away from the gap until its bbox touches the grid edge. Sprites stay fixed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _object_info(g)
    sprites = _sprite_info(g)
    assert objects, "expected at least one blue object"
    assert sprites, "expected at least one sprite"

    out = g.copy()
    out[g == BLUE] = BG
    occupied = out != BG

    for obj in objects:
        matches = [sp for sp in sprites if obj["gap_shape"] in sp["variants"]]
        assert len(matches) == 1, "each gap must match exactly one sprite up to rotation/flip"
        color = int(matches[0]["color"])
        patch = _render_object_patch(obj["h"], obj["w"], obj["side"], color)
        tr, tc = _moved_origin(obj, g.shape)
        target = out[tr : tr + obj["h"], tc : tc + obj["w"]]
        assert target.shape == patch.shape
        assert not bool(np.any(occupied[tr : tr + obj["h"], tc : tc + obj["w"]] & (patch != BG))), "moved object overlap"
        target[patch != BG] = patch[patch != BG]
        occupied[tr : tr + obj["h"], tc : tc + obj["w"]] |= patch != BG

    return out


def _box_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0] - pad
        or b[2] + pad < a[0] - pad
        or a[3] + pad < b[1] - pad
        or b[3] + pad < a[1] - pad
    )


def _make_object_patch(side: str, h: int, w: int, gap_len: int) -> np.ndarray:
    patch = np.full((h, w), BLUE, dtype=int)
    if side in ("top", "bottom"):
        start = random.randint(1, w - gap_len - 1)
        row = 0 if side == "top" else h - 1
        patch[row, start : start + gap_len] = BG
    else:
        start = random.randint(1, h - gap_len - 1)
        col = 0 if side == "left" else w - 1
        patch[start : start + gap_len, col] = BG
    return patch


def _make_sprite_patch(length: int, horizontal: bool, color: int) -> np.ndarray:
    if horizontal:
        patch = np.full((1, length), color, dtype=int)
    else:
        patch = np.full((length, 1), color, dtype=int)
    return patch


def _place_final_boxes(specs: list[dict], h: int, w: int) -> bool:
    placed: list[tuple[int, int, int, int]] = []
    for spec in specs:
        oh = spec["obj_h"]
        ow = spec["obj_w"]
        side = spec["side"]
        ok = []
        if side == "top":
            rr = h - oh
            for cc in range(0, w - ow + 1):
                ok.append((rr, cc, rr + oh - 1, cc + ow - 1))
        elif side == "bottom":
            rr = 0
            for cc in range(0, w - ow + 1):
                ok.append((rr, cc, rr + oh - 1, cc + ow - 1))
        elif side == "left":
            cc = w - ow
            for rr in range(0, h - oh + 1):
                ok.append((rr, cc, rr + oh - 1, cc + ow - 1))
        else:
            cc = 0
            for rr in range(0, h - oh + 1):
                ok.append((rr, cc, rr + oh - 1, cc + ow - 1))
        random.shuffle(ok)
        chosen = None
        for box in ok:
            if any(_box_overlap(box, prev, pad=1) for prev in placed):
                continue
            chosen = box
            break
        if chosen is None:
            return False
        placed.append(chosen)
        spec["final_box"] = chosen
    return True


def _place_input_boxes(specs: list[dict]) -> bool:
    placed: list[tuple[int, int, int, int]] = []
    for spec in specs:
        fr0, fc0, fr1, fc1 = spec["final_box"]
        oh = spec["obj_h"]
        ow = spec["obj_w"]
        side = spec["side"]
        choices = []
        if side == "top":
            for rr in range(0, fr0 - 1):
                choices.append((rr, fc0, rr + oh - 1, fc0 + ow - 1))
        elif side == "bottom":
            for rr in range(fr0 + 2, spec["grid_h"] - oh + 1):
                choices.append((rr, fc0, rr + oh - 1, fc0 + ow - 1))
        elif side == "left":
            for cc in range(0, fc0 - 1):
                choices.append((fr0, cc, fr0 + oh - 1, cc + ow - 1))
        else:
            for cc in range(fc0 + 2, spec["grid_w"] - ow + 1):
                choices.append((fr0, cc, fr0 + oh - 1, cc + ow - 1))
        random.shuffle(choices)
        chosen = None
        for box in choices:
            if any(_box_overlap(box, prev, pad=1) for prev in placed):
                continue
            chosen = box
            break
        if chosen is None:
            return False
        placed.append(chosen)
        spec["input_box"] = chosen
    return True


def _place_sprites(specs: list[dict], h: int, w: int) -> bool:
    occupied = [spec["input_box"] for spec in specs] + [spec["final_box"] for spec in specs]
    sprite_boxes: list[tuple[int, int, int, int]] = []
    for spec in specs:
        sh, sw = spec["sprite_patch"].shape
        choices = []
        for r0 in range(0, h - sh + 1):
            for c0 in range(0, w - sw + 1):
                box = (r0, c0, r0 + sh - 1, c0 + sw - 1)
                choices.append(box)
        random.shuffle(choices)
        chosen = None
        for box in choices:
            if any(_box_overlap(box, other, pad=1) for other in occupied):
                continue
            if any(_box_overlap(box, other, pad=1) for other in sprite_boxes):
                continue
            chosen = box
            break
        if chosen is None:
            return False
        sprite_boxes.append(chosen)
        spec["sprite_box"] = chosen
    return True


def _draw_patch(grid: np.ndarray, patch: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    target = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert target.shape == patch.shape
    assert not bool(np.any((target != BG) & (patch != BG)))
    target[patch != BG] = patch[patch != BG]


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    grid_h = random.randint(18, 28)
    grid_w = random.randint(18, 28)
    n_objects = random.randint(1, 3)

    lengths = [1, 2, 3]
    random.shuffle(lengths)
    chosen_lengths = sorted(lengths[:n_objects])
    colors = random.sample(SPRITE_COLORS, n_objects)

    specs = []
    sides = [random.choice(SIDES) for _ in range(n_objects)]
    for idx, gap_len in enumerate(chosen_lengths):
        side = sides[idx]
        if side in ("top", "bottom"):
            obj_h = random.randint(4, 7)
            obj_w = random.randint(max(5, gap_len + 3), 9)
        else:
            obj_h = random.randint(max(5, gap_len + 3), 9)
            obj_w = random.randint(4, 7)
        obj_patch = _make_object_patch(side, obj_h, obj_w, gap_len)
        sprite_patch = _make_sprite_patch(gap_len, bool(random.getrandbits(1)), colors[idx])
        specs.append(
            {
                "side": side,
                "gap_len": gap_len,
                "color": colors[idx],
                "obj_h": obj_h,
                "obj_w": obj_w,
                "obj_patch": obj_patch,
                "sprite_patch": sprite_patch,
                "grid_h": grid_h,
                "grid_w": grid_w,
            }
        )

    if not _place_final_boxes(specs, grid_h, grid_w):
        return None
    if not _place_input_boxes(specs):
        return None
    if not _place_sprites(specs, grid_h, grid_w):
        return None

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = np.full((grid_h, grid_w), BG, dtype=int)

    for spec in specs:
        _draw_patch(inp, spec["obj_patch"], spec["input_box"])
        _draw_patch(inp, spec["sprite_patch"], spec["sprite_box"])
        _draw_patch(out, spec["sprite_patch"], spec["sprite_box"])
        moved = _render_object_patch(spec["obj_h"], spec["obj_w"], spec["side"], spec["color"])
        _draw_patch(out, moved, spec["final_box"])

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sample = _sample_case()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid ARC sample")
