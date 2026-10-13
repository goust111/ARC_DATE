import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = tuple(range(10))
GEN_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_true(mask: np.ndarray) -> np.ndarray | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _has_interior(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        if all(0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc] for dr, dc in DIR4):
            return True
    return False


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    boundary: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                boundary.append((int(r), int(c)))
                break
    return boundary


def _frame_holes(grid: np.ndarray, bg: int, comp: dict) -> list[dict] | None:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not (
        np.all(sub[0, :] == color)
        and np.all(sub[-1, :] == color)
        and np.all(sub[:, 0] == color)
        and np.all(sub[:, -1] == color)
    ):
        return None
    if not set(map(int, np.unique(sub))).issubset({bg, color}):
        return None

    holes: list[dict] = []
    hole_mask = sub == bg
    for coords in _components(hole_mask):
        if (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == sub.shape[0] - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == sub.shape[1] - 1)
        ):
            continue
        hr0, hc0, hr1, hc1 = _bbox(coords)
        local = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
        local[coords[:, 0] - hr0, coords[:, 1] - hc0] = True
        holes.append(
            {
                "top": int(hr0),
                "left": int(hc0),
                "mask": local,
                "shape": _shape_signature(local),
            }
        )
    return holes or None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    components: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(g == color):
            components.append({"color": color, "coords": coords, "bbox": _bbox(coords)})

    frames: list[dict] = []
    sprites: list[dict] = []
    for comp in components:
        holes = _frame_holes(g, bg, comp)
        if holes is None:
            sprites.append(comp)
        else:
            comp = dict(comp)
            comp["holes"] = holes
            frames.append(comp)

    assert frames, "expected at least one frame"
    assert sprites, "expected at least one external sprite"

    shape_to_sprite: dict[tuple[tuple[int, int], ...], dict] = {}
    for comp in sprites:
        r0, c0, r1, c1 = comp["bbox"]
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        color = int(comp["color"])
        assert set(map(int, np.unique(sub))).issubset({bg, color})
        mask = sub == color
        shape = _shape_signature(mask)
        assert shape not in shape_to_sprite, "sprite shapes must be unique"
        shape_to_sprite[shape] = {"color": color, "coords": comp["coords"]}

    out = g.copy()
    for comp in sprites:
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    for frame in frames:
        fr0, fc0, _, _ = frame["bbox"]
        for hole in frame["holes"]:
            shape = hole["shape"]
            assert shape in shape_to_sprite, "every hole needs a matching sprite"
            used_shapes.add(shape)
            color = int(shape_to_sprite[shape]["color"])
            for dr, dc in _boundary_cells(hole["mask"]):
                out[fr0 + hole["top"] + dr, fc0 + hole["left"] + dc] = color

    assert used_shapes == set(shape_to_sprite), "every sprite should be used exactly once"
    return out


def _random_hole_mask(
    rng: random.Random, used_shapes: set[tuple[tuple[int, int], ...]]
) -> np.ndarray | None:
    for _ in range(200):
        h = rng.randint(3, 6)
        w = rng.randint(3, 6)
        mask = np.ones((h, w), dtype=bool)
        perimeter = [(r, c) for r in range(h) for c in range(w) if r in {0, h - 1} or c in {0, w - 1}]
        rng.shuffle(perimeter)
        target = rng.randint(1, max(1, min(len(perimeter) // 2, h + w - 3)))
        removed = 0
        for r, c in perimeter:
            if removed >= target or not mask[r, c]:
                continue
            trial = mask.copy()
            trial[r, c] = False
            cropped = _crop_true(trial)
            if cropped is None or np.all(cropped):
                continue
            if not _is_connected(cropped) or not _has_interior(cropped):
                continue
            mask = trial
            removed += 1
        cropped = _crop_true(mask)
        if cropped is None or removed == 0 or np.all(cropped):
            continue
        shape = _shape_signature(cropped)
        if shape in used_shapes:
            continue
        used_shapes.add(shape)
        return cropped
    return None


def _build_frame_patch(frame_color: int, bg: int, hole_specs: list[dict], rng: random.Random) -> dict:
    specs = [dict(spec) for spec in hole_specs]
    rng.shuffle(specs)
    orientation = rng.choice(["row", "col"])
    gap = rng.randint(1, 2)

    if orientation == "row":
        inner_h = max(spec["mask"].shape[0] for spec in specs) + rng.randint(0, 1) + rng.randint(0, 1)
        inner_w = sum(spec["mask"].shape[1] for spec in specs) + gap * (len(specs) - 1) + rng.randint(0, 1) + rng.randint(0, 1)
        patch = np.full((inner_h + 2, inner_w + 2), frame_color, dtype=int)
        cursor = 1 + rng.randint(0, 1)
        placed_holes = []
        for spec in specs:
            mask = spec["mask"]
            hh, hw = mask.shape
            top = 1 + rng.randint(0, inner_h - hh)
            left = cursor
            window = patch[top : top + hh, left : left + hw]
            window[mask] = bg
            placed_holes.append({"top": top, "left": left, "mask": mask, "sprite_color": spec["sprite_color"]})
            cursor += hw + gap
    else:
        inner_h = sum(spec["mask"].shape[0] for spec in specs) + gap * (len(specs) - 1) + rng.randint(0, 1) + rng.randint(0, 1)
        inner_w = max(spec["mask"].shape[1] for spec in specs) + rng.randint(0, 1) + rng.randint(0, 1)
        patch = np.full((inner_h + 2, inner_w + 2), frame_color, dtype=int)
        cursor = 1 + rng.randint(0, 1)
        placed_holes = []
        for spec in specs:
            mask = spec["mask"]
            hh, hw = mask.shape
            top = cursor
            left = 1 + rng.randint(0, inner_w - hw)
            window = patch[top : top + hh, left : left + hw]
            window[mask] = bg
            placed_holes.append({"top": top, "left": left, "mask": mask, "sprite_color": spec["sprite_color"]})
            cursor += hh + gap

    return {"patch": patch, "holes": placed_holes, "color": frame_color}


def _build_sprite_patch(mask: np.ndarray, color: int, bg: int) -> dict:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = color
    return {"patch": patch, "mask": mask, "color": color}


def _place_object(
    canvas: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    rng: random.Random,
    bg: int,
) -> tuple[int, int] | None:
    h, w = patch.shape
    H, W = canvas.shape
    if h + 2 > H or w + 2 > W:
        return None
    positions = [(r, c) for r in range(1, H - h) for c in range(1, W - w)]
    rng.shuffle(positions)
    for r, c in positions:
        if blocked[r : r + h, c : c + w].any():
            continue
        window = canvas[r : r + h, c : c + w]
        mask = patch != bg
        window[mask] = patch[mask]
        blocked[r - 1 : r + h + 1, c - 1 : c + w + 1] = True
        return int(r), int(c)
    return None


def _build_expected(inp: np.ndarray, bg: int, frames: list[dict], sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    for sprite in sprites:
        top = int(sprite["top"])
        left = int(sprite["left"])
        mask = sprite["mask"]
        h, w = mask.shape
        window = out[top : top + h, left : left + w]
        window[mask] = bg

    for frame in frames:
        ftop = int(frame["top"])
        fleft = int(frame["left"])
        for hole in frame["holes"]:
            color = int(hole["sprite_color"])
            top = ftop + int(hole["top"])
            left = fleft + int(hole["left"])
            for dr, dc in _boundary_cells(hole["mask"]):
                out[top + dr, left + dc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = rng.choice(COLOR_POOL)
    frame_count = rng.randint(1, 3)
    hole_counts = [1] * frame_count
    extra_holes = rng.randint(0, 3)
    for _ in range(extra_holes):
        candidates = [idx for idx, count in enumerate(hole_counts) if count < 3]
        if not candidates:
            break
        hole_counts[rng.choice(candidates)] += 1
    total_holes = sum(hole_counts)

    if frame_count + total_holes > len(COLOR_POOL) - 1:
        return None

    palette = [color for color in COLOR_POOL if color != bg]
    rng.shuffle(palette)
    frame_colors = palette[:frame_count]
    sprite_colors = palette[frame_count : frame_count + total_holes]

    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    hole_masks: list[np.ndarray] = []
    for _ in range(total_holes):
        mask = _random_hole_mask(rng, used_shapes)
        if mask is None:
            return None
        hole_masks.append(mask)

    frame_objects: list[dict] = []
    sprite_objects: list[dict] = []
    cursor = 0
    for frame_color, hole_count in zip(frame_colors, hole_counts):
        hole_specs = []
        for _ in range(hole_count):
            mask = hole_masks[cursor]
            sprite_color = sprite_colors[cursor]
            hole_specs.append({"mask": mask, "sprite_color": sprite_color})
            sprite_objects.append(_build_sprite_patch(mask, sprite_color, bg))
            cursor += 1
        frame_objects.append(_build_frame_patch(frame_color, bg, hole_specs, rng))

    all_patches = [obj["patch"] for obj in frame_objects + sprite_objects]
    max_h = max(p.shape[0] for p in all_patches)
    max_w = max(p.shape[1] for p in all_patches)
    total_bbox_area = sum(p.shape[0] * p.shape[1] for p in all_patches)
    base_side = max(24, int((total_bbox_area * 2.2) ** 0.5), max_h + 4, max_w + 4)
    if base_side > 36:
        return None
    H = rng.randint(base_side, min(36, base_side + 6))
    W = rng.randint(base_side, min(36, base_side + 6))

    inp = np.full((H, W), bg, dtype=int)
    blocked = np.zeros((H, W), dtype=bool)

    for obj in sorted(frame_objects, key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True):
        pos = _place_object(inp, blocked, obj["patch"], rng, bg)
        if pos is None:
            return None
        obj["top"], obj["left"] = pos

    for obj in sorted(sprite_objects, key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True):
        pos = _place_object(inp, blocked, obj["patch"], rng, bg)
        if pos is None:
            return None
        obj["top"], obj["left"] = pos

    counts = {int(color): int(np.sum(inp == color)) for color in np.unique(inp)}
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return None

    expected = _build_expected(inp, bg, frame_objects, sprite_objects)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
