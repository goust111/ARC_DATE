from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 8
BAR_COLORS = (2, 7)
FRAME_COLORS = (1, 3, 4, 6)
SPRITE_COLOR_POOL = (0, 1, 3, 4, 5, 6, 9)
FRAME_SIZE = 11
MAX_GEN_ATTEMPTS = 400
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _label_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _enclosed_background(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    bg_mask = ~mask
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if bg_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return bg_mask & ~outside


def _hole_count(mask: np.ndarray) -> int:
    return len(_label_components(_enclosed_background(mask)))


def _sprite_templates() -> dict[int, np.ndarray]:
    raw = {
        1: [
            "111",
            "101",
            "110",
        ],
        2: [
            "11111",
            "10101",
            "11110",
        ],
        3: [
            "11111",
            "10101",
            "11111",
            "10111",
            "11110",
        ],
        4: [
            "11111",
            "10101",
            "11111",
            "10101",
            "11110",
        ],
    }
    out: dict[int, np.ndarray] = {}
    for holes, rows in raw.items():
        arr = np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)
        assert _hole_count(arr) == holes
        assert not np.array_equal(arr, np.fliplr(arr))
        out[holes] = arr
    return out


SPRITE_TEMPLATES = _sprite_templates()


def _make_frame_mask(hole_count: int) -> np.ndarray:
    mask = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    pockets = {
        "tl": [(1, 2), (2, 2), (2, 1)],
        "tr": [(1, 8), (2, 8), (2, 9)],
        "bl": [(8, 1), (8, 2), (9, 2)],
    }
    order = ("tl", "tr", "bl")
    for key in order[: hole_count - 1]:
        for r, c in pockets[key]:
            mask[r, c] = True
    assert _hole_count(mask) == hole_count
    return mask


def _find_monochrome_components(grid: np.ndarray, color: int) -> list[dict]:
    comps = []
    for coords in _label_components(grid == int(color)):
        comp_bbox = _bbox(coords)
        comps.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": comp_bbox,
                "mask": _crop_mask(grid.shape, coords),
            }
        )
    return comps


def _extract_component_mask(comp: dict) -> np.ndarray:
    return _crop_mask(comp["coords"].shape, comp["coords"])


def _is_frame_component(comp: dict) -> bool:
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 7 or w < 7:
        return False
    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if not local[0, :].all() or not local[-1, :].all() or not local[:, 0].all() or not local[:, -1].all():
        return False
    return _hole_count(local) >= 1


def _component_local_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[comp["coords"][:, 0] - r0, comp["coords"][:, 1] - c0] = True
    return out


def _horizontal_bar_pairs(grid: np.ndarray) -> list[dict]:
    bars = []
    for r in range(grid.shape[0]):
        c = 0
        while c < grid.shape[1] - 1:
            a = int(grid[r, c])
            b = int(grid[r, c + 1])
            if {a, b} == set(BAR_COLORS) and a != b:
                left_ok = c == 0 or int(grid[r, c - 1]) == BG
                right_ok = c + 2 >= grid.shape[1] or int(grid[r, c + 2]) == BG
                if left_ok and right_ok:
                    bars.append({"row": int(r), "c0": int(c), "order": (a, b)})
                    c += 2
                    continue
            c += 1
    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each sprite to the frame with the same enclosed-hole count, orient it from the two-color bar, and nest it inside the frame.

    Input structure:
    - The background is azure (8), with one reference two-color bar near the top-left using red (2) and orange (7).
    - The upper area contains several monochrome hollow frame objects whose internal walls may split the enclosed background into different numbers of holes.
    - The lower area contains the same number of monochrome sprite objects; each sprite has its own enclosed holes and a touching horizontal two-color bar directly above its top row.

    Rule:
    - Count 4-connected enclosed background components inside every frame and every sprite.
    - Pair sprites to frames by equal hole count.
    - Compare each sprite's local two-color bar with the reference bar: matching order keeps the sprite, reversed order flips the sprite horizontally.
    - Remove all loose sprites and bars, then place each oriented sprite into the matched frame, centered inside the frame interior with top-left bias on odd slack; recolor sprite-hole pixels to the frame color.

    Key insight:
    - Hole count determines which sprite belongs in which frame, while the two-color bar determines only left-right orientation, so both independent clues are required.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        components.extend(_find_monochrome_components(g, color))

    frames = [comp for comp in components if comp["color"] not in BAR_COLORS and _is_frame_component(comp)]
    assert 3 <= len(frames) <= 4
    frame_by_holes: dict[int, dict] = {}
    for frame in frames:
        holes = _hole_count(_component_local_mask(frame))
        assert holes not in frame_by_holes
        frame["holes"] = int(holes)
        frame_by_holes[int(holes)] = frame

    frame_cells = np.zeros(g.shape, dtype=bool)
    for frame in frames:
        rr, cc = frame["coords"][:, 0], frame["coords"][:, 1]
        frame_cells[rr, cc] = True

    sprites = []
    for comp in components:
        if comp["color"] in BAR_COLORS:
            continue
        if any(np.array_equal(comp["coords"], frame["coords"]) for frame in frames):
            continue
        if frame_cells[comp["coords"][:, 0], comp["coords"][:, 1]].any():
            continue
        local = _component_local_mask(comp)
        holes = _hole_count(local)
        if holes in frame_by_holes:
            comp["holes"] = int(holes)
            comp["local_mask"] = local
            sprites.append(comp)

    assert len(sprites) == len(frames)
    sprite_by_holes = {}
    for sprite in sprites:
        assert sprite["holes"] not in sprite_by_holes
        sprite_by_holes[int(sprite["holes"])] = sprite

    bars = _horizontal_bar_pairs(g)
    assert len(bars) == len(sprites) + 1
    reference_bar = min(bars, key=lambda b: (b["row"], b["c0"]))

    used_bars: set[tuple[int, int]] = {(reference_bar["row"], reference_bar["c0"])}
    for sprite in sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        bar_matches = [
            bar
            for bar in bars
            if (bar["row"], bar["c0"]) not in used_bars
            and bar["row"] == r0 - 1
            and c0 <= bar["c0"] <= c1
        ]
        assert len(bar_matches) == 1
        sprite["bar"] = bar_matches[0]
        used_bars.add((bar_matches[0]["row"], bar_matches[0]["c0"]))

    out = g.copy()
    for bar in bars:
        out[bar["row"], bar["c0"] : bar["c0"] + 2] = BG
    for sprite in sprites:
        rr, cc = sprite["coords"][:, 0], sprite["coords"][:, 1]
        out[rr, cc] = BG

    for holes, frame in frame_by_holes.items():
        sprite = sprite_by_holes[holes]
        sprite_mask = sprite["local_mask"].copy()
        sprite_holes = _enclosed_background(sprite_mask)
        if sprite["bar"]["order"] != reference_bar["order"]:
            sprite_mask = np.fliplr(sprite_mask)
            sprite_holes = np.fliplr(sprite_holes)

        frame_color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        sh, sw = sprite_mask.shape
        assert sh <= inner_h and sw <= inner_w
        dst_r = r0 + 1 + (inner_h - sh) // 2
        dst_c = c0 + 1 + (inner_w - sw) // 2

        patch = np.full(sprite_mask.shape, BG, dtype=int)
        patch[sprite_mask] = int(sprite["color"])
        patch[sprite_holes] = frame_color

        area = out[dst_r : dst_r + sh, dst_c : dst_c + sw]
        assert not np.any((patch != BG) & (area != BG))
        area[patch != BG] = patch[patch != BG]

    return out


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    view = grid[top : top + h, left : left + w]
    view[mask] = int(color)


def _paint_frame(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    _paint_mask(grid, mask, top, left, color)


def _random_frame_layout(count: int) -> tuple[int, int, list[tuple[int, int]]]:
    if count == 4:
        grid_h = 30
        grid_w = random.randint(24, 30)
        x0 = random.randint(0, grid_w - 24)
        positions = [(1, x0), (1, x0 + 13), (12, x0), (12, x0 + 13)]
        return grid_h, grid_w, positions

    layout_choices = [
        (29, 24, [(1, 1), (1, 13), (12, 7)]),
        (30, 25, [(1, 1), (1, 13), (12, 1)]),
        (30, 27, [(1, 3), (1, 15), (12, 9)]),
    ]
    grid_h, min_w, positions = random.choice(layout_choices)
    grid_w = random.randint(min_w, 30)
    shift = random.randint(0, grid_w - min_w)
    shifted = [(r, c + shift) for r, c in positions]
    return grid_h, grid_w, shifted


def _sprite_patch(mask: np.ndarray, sprite_color: int, frame_color: int) -> np.ndarray:
    holes = _enclosed_background(mask)
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(sprite_color)
    patch[holes] = int(frame_color)
    return patch


def _reference_order() -> tuple[int, int]:
    return random.choice([(2, 7), (7, 2)])


def _can_place(blocked: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(blocked.shape[0], top + h + pad)
    c1 = min(blocked.shape[1], left + w + pad)
    return not blocked[r0:r1, c0:c1].any()


def _mark(blocked: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    blocked[top : top + h, left : left + w] = True


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    frame_holes: list[int],
    sprite_info: list[dict],
    reference: tuple[int, int],
) -> None:
    assert len(set(frame_holes)) == len(frame_holes)
    assert sorted(frame_holes) == sorted(info["holes"] for info in sprite_info)
    assert any(info["bar"] == reference for info in sprite_info)
    assert any(info["bar"] != reference for info in sprite_info)
    for info in sprite_info:
        mask = info["mask"]
        assert not np.array_equal(mask, np.fliplr(mask))
        oriented = mask if info["bar"] == reference else np.fliplr(mask)
        wrong = np.fliplr(oriented)
        assert not np.array_equal(oriented, wrong)
    actual = main(inp)
    assert np.array_equal(actual, out)
    toggled = inp.copy()
    target = sprite_info[0]
    bar_r, bar_c = target["bar_pos"]
    toggled[bar_r, bar_c] = target["bar"][1]
    toggled[bar_r, bar_c + 1] = target["bar"][0]
    assert not np.array_equal(main(toggled), out)


def _gen_attempt() -> np.ndarray | None:
    count = random.choice([3, 4])
    frame_holes = list(range(1, count + 1))
    random.shuffle(frame_holes)
    reference = _reference_order()
    grid_h, grid_w, frame_positions = _random_frame_layout(count)

    frame_colors = random.sample(FRAME_COLORS, count)
    sprite_colors = random.sample(SPRITE_COLOR_POOL, count)
    if count == 4 and len(set(sprite_colors)) < count:
        return None

    out = np.full((grid_h, grid_w), BG, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    frame_specs = []
    for holes, color, (top, left) in zip(frame_holes, frame_colors, frame_positions):
        frame_mask = _make_frame_mask(holes)
        _paint_frame(out, frame_mask, top, left, color)
        _mark(blocked, top, left, FRAME_SIZE, FRAME_SIZE)
        frame_specs.append(
            {
                "holes": holes,
                "color": int(color),
                "top": int(top),
                "left": int(left),
                "mask": frame_mask,
            }
        )

    sprite_order = frame_holes[:]
    random.shuffle(sprite_order)
    sprite_specs = []
    for idx, holes in enumerate(sprite_order):
        sprite_mask = SPRITE_TEMPLATES[holes].copy()
        same_orientation = idx % 2 == 0
        if count == 3 and idx == 2 and all(spec["bar"] == reference for spec in sprite_specs):
            same_orientation = False
        bar = reference if same_orientation else (reference[1], reference[0])
        frame = next(spec for spec in frame_specs if spec["holes"] == holes)
        patch = _sprite_patch(sprite_mask if bar == reference else np.fliplr(sprite_mask), sprite_colors[idx], frame["color"])
        inner_top = frame["top"] + 1 + (FRAME_SIZE - 2 - patch.shape[0]) // 2
        inner_left = frame["left"] + 1 + (FRAME_SIZE - 2 - patch.shape[1]) // 2
        area = out[inner_top : inner_top + patch.shape[0], inner_left : inner_left + patch.shape[1]]
        area[patch != BG] = patch[patch != BG]
        sprite_specs.append(
            {
                "holes": holes,
                "color": int(sprite_colors[idx]),
                "mask": sprite_mask,
                "bar": bar,
            }
        )

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    for spec in frame_specs:
        _paint_frame(inp, spec["mask"], spec["top"], spec["left"], spec["color"])

    ref_col = random.randint(0, min(2, grid_w - 2))
    inp[0, ref_col] = reference[0]
    inp[0, ref_col + 1] = reference[1]
    blocked_input = blocked.copy()
    _mark(blocked_input, 0, ref_col, 1, 2)

    lower_top = max(top + FRAME_SIZE for top, _ in frame_positions)
    start_row = lower_top
    if start_row + 6 > grid_h:
        return None

    random.shuffle(sprite_specs)
    for spec in sprite_specs:
        sprite_mask = spec["mask"]
        sh, sw = sprite_mask.shape
        total_h = sh + 1
        candidates = []
        for r in range(start_row, grid_h - total_h + 1):
            for c in range(0, grid_w - sw + 1):
                if _can_place(blocked_input, r, c, sh, sw, pad=1) and _can_place(blocked_input, r - 1, c, 1, 2, pad=0):
                    candidates.append((r, c))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        _paint_mask(inp, sprite_mask, top, left, spec["color"])
        inp[top - 1, left] = spec["bar"][0]
        inp[top - 1, left + 1] = spec["bar"][1]
        _mark(blocked_input, top, left, sh, sw)
        _mark(blocked_input, top - 1, left, 1, 2)
        spec["bar_pos"] = (top - 1, left)

    _semantic_checks(inp, out, frame_holes, sprite_specs, reference)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
