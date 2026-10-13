from __future__ import annotations

import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
ALL_COLORS = tuple(range(10))

GEN_RETRIES = 400
PLACEMENT_RETRIES = 50
CANVAS_MIN = 14
CANVAS_MAX = 30


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize_coords(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return norm, int(r1 - r0 + 1), int(c1 - c0 + 1)


def _shape_key_from_mask(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    key, _, _ = _normalize_coords(coords)
    return key


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    key, h, w = _normalize_coords(coords)
    out = np.zeros((h, w), dtype=bool)
    for r, c in key:
        out[r, c] = True
    return out


def _is_rect_frame_component(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> bool:
    norm, h, w = _normalize_coords(coords)
    if h < 3 or w < 3 or len(norm) != 2 * (h + w) - 4:
        return False
    pts = set(norm)
    for r, c in pts:
        if r not in (0, h - 1) and c not in (0, w - 1):
            return False
    for c in range(w):
        if (0, c) not in pts or (h - 1, c) not in pts:
            return False
    for r in range(h):
        if (r, 0) not in pts or (r, w - 1) not in pts:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each framed gap with the matching outside sprite, recolored to the frame border.

    Input structure:
    - The background is the most common color.
    - Several one-pixel-thick monochrome rectangular frames appear in the grid; each frame encloses a multicolor interior pattern.
    - Inside every frame interior, exactly one 4-connected background region forms the missing gap shape.
    - Outside all frame bounding boxes, several monochrome sprite components are scattered; matched sprites share shape with a frame gap under translation only.

    Rule:
    - Detect every rectangular frame and read its border color.
    - For each frame, extract the unique background gap shape inside the frame interior.
    - Among the outside sprites, find the unused sprite with exactly the same normalized shape, without rotation or reflection.
    - Remove that sprite from its outside location and fill the frame gap with the frame border color; leave unmatched sprites unchanged.

    Key insight:
    - The sprite's own color is irrelevant noise: only exact shape selects the piece, while the destination frame supplies the final fill color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    frames: list[dict[str, object]] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        for comp in _components(g == color):
            if not _is_rect_frame_component(comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            frames.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))  # type: ignore[index]
    assert frames, "expected at least one rectangular frame"

    outside = np.ones(g.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        outside[r0 : r1 + 1, c0 : c1 + 1] = False

    sprites: list[dict[str, object]] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        for comp in _components((g == color) & outside):
            key, _, _ = _normalize_coords(comp)
            sprites.append({"color": int(color), "coords": comp, "key": key})

    shape_to_sprites: dict[tuple[tuple[int, int], ...], list[int]] = {}
    for idx, sprite in enumerate(sprites):
        shape_to_sprites.setdefault(sprite["key"], []).append(idx)  # type: ignore[index]

    used: set[int] = set()
    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        frame_color = int(frame["color"])  # type: ignore[arg-type]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        holes = _components(interior == bg)
        assert len(holes) == 1, "each frame interior must have exactly one background gap"
        hole = holes[0]
        hole_key, _, _ = _normalize_coords(hole)
        candidates = [idx for idx in shape_to_sprites.get(hole_key, []) if idx not in used]
        assert len(candidates) == 1, "expected a unique matching outside sprite for every frame"
        sprite_idx = candidates[0]
        used.add(sprite_idx)

        for rr, cc in hole:
            out[r0 + 1 + rr, c0 + 1 + cc] = frame_color
        for rr, cc in sprites[sprite_idx]["coords"]:  # type: ignore[index]
            out[rr, cc] = bg

    return out


def _grow_connected_mask(h: int, w: int, target: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    chosen = {start}
    frontier = set()
    for dr, dc in DIR4:
        nr = start[0] + dr
        nc = start[1] + dc
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))
    while len(chosen) < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.discard(cell)
        if cell in chosen:
            continue
        chosen.add(cell)
        mask[cell] = True
        for dr, dc in DIR4:
            nr = cell[0] + dr
            nc = cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in chosen:
                frontier.add((nr, nc))
    if len(chosen) != target:
        return None
    return mask


def _random_irregular_mask(
    max_h: int,
    max_w: int,
    forbidden: set[tuple[tuple[int, int], ...]],
    max_cells: int | None = None,
) -> np.ndarray | None:
    if max_h < 2 or max_w < 2:
        return None
    for _ in range(200):
        h = random.randint(2, max_h)
        w = random.randint(2, max_w)
        area_cap = h * w - 1
        if max_cells is not None:
            area_cap = min(area_cap, int(max_cells))
        if area_cap < 3:
            continue
        target = random.randint(3, max(3, area_cap))
        mask = _grow_connected_mask(h, w, target)
        if mask is None:
            continue
        mask = _tight_mask(mask)
        mh, mw = mask.shape
        if mh < 2 or mw < 2:
            continue
        if int(mask.sum()) == int(mh * mw):
            continue
        if _is_rect_frame_component([(int(r), int(c)) for r, c in np.argwhere(mask)]):
            continue
        key = _shape_key_from_mask(mask)
        if key in forbidden:
            continue
        return mask
    return None


def _paint_pattern(
    interior_h: int,
    interior_w: int,
    hole_mask: np.ndarray,
    bg: int,
    frame_color: int,
) -> np.ndarray:
    interior = np.full((interior_h, interior_w), bg, dtype=int)
    cells = [(r, c) for r in range(interior_h) for c in range(interior_w) if not hole_mask[r, c]]
    palette = [c for c in ALL_COLORS if c not in {bg, frame_color}]
    color_count = random.randint(2, min(3, len(cells)))
    colors = random.sample(palette, color_count)
    seeds = random.sample(cells, color_count)
    for r, c in cells:
        best = min(
            range(color_count),
            key=lambda i: (abs(r - seeds[i][0]) + abs(c - seeds[i][1]), i),
        )
        interior[r, c] = colors[best]
    return interior


def _frame_patch(
    outer_h: int,
    outer_w: int,
    bg: int,
    frame_color: int,
    used_shapes: set[tuple[tuple[int, int], ...]],
) -> tuple[dict[str, object], dict[str, object]] | None:
    interior_h = outer_h - 2
    interior_w = outer_w - 2
    hole_shape = _random_irregular_mask(
        interior_h,
        interior_w,
        used_shapes,
        max_cells=interior_h * interior_w - 2,
    )
    if hole_shape is None:
        return None
    hole_key = _shape_key_from_mask(hole_shape)
    used_shapes.add(hole_key)

    hole_h, hole_w = hole_shape.shape
    top = random.randint(0, interior_h - hole_h)
    left = random.randint(0, interior_w - hole_w)
    hole_mask = np.zeros((interior_h, interior_w), dtype=bool)
    hole_mask[top : top + hole_h, left : left + hole_w] = hole_shape

    patch = np.full((outer_h, outer_w), bg, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    patch[1:-1, 1:-1] = _paint_pattern(interior_h, interior_w, hole_mask, bg, frame_color)
    patch[1:-1, 1:-1][hole_mask] = bg

    frame_obj = {
        "kind": "frame",
        "patch": patch,
        "frame_color": int(frame_color),
        "hole_mask": hole_mask,
    }
    sprite_patch = np.full((hole_h, hole_w), bg, dtype=int)
    sprite_obj = {
        "kind": "sprite",
        "patch": sprite_patch,
        "matched": True,
        "shape_key": hole_key,
    }
    sprite_patch[hole_shape] = -1
    return frame_obj, sprite_obj


def _place_objects(objects: list[dict[str, object]], height: int, width: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(objects)
    order = sorted(
        range(len(objects)),
        key=lambda idx: (
            objects[idx]["patch"].shape[0] * objects[idx]["patch"].shape[1],  # type: ignore[index]
            objects[idx]["patch"].shape[0],  # type: ignore[index]
            objects[idx]["patch"].shape[1],  # type: ignore[index]
        ),
        reverse=True,
    )
    for idx in order:
        patch = objects[idx]["patch"]  # type: ignore[index]
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r in range(1, height - ph):
            for c in range(1, width - pw):
                r0 = r - 1
                c0 = c - 1
                r1 = r + ph + 1
                c1 = c + pw + 1
                if occupied[r0:r1, c0:c1].any():
                    continue
                candidates.append((int(r), int(c)))
        if not candidates:
            return None
        pos = random.choice(candidates)
        positions[idx] = pos
        occupied[pos[0] : pos[0] + ph, pos[1] : pos[1] + pw] = True
    return [pos for pos in positions if pos is not None]


def _sample_canvas(objects: list[dict[str, object]], bg: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    max_h = max(obj["patch"].shape[0] for obj in objects) + 2  # type: ignore[index]
    max_w = max(obj["patch"].shape[1] for obj in objects) + 2  # type: ignore[index]
    bbox_area = sum(obj["patch"].shape[0] * obj["patch"].shape[1] for obj in objects)  # type: ignore[index]
    fg_cells = sum(int(np.count_nonzero(obj["patch"] != bg)) for obj in objects)  # type: ignore[index]
    target_area = max(CANVAS_MIN * CANVAS_MIN, bbox_area * 2, fg_cells * 3)
    if target_area > CANVAS_MAX * CANVAS_MAX:
        return None

    for _ in range(PLACEMENT_RETRIES):
        height = random.randint(max(CANVAS_MIN, max_h), CANVAS_MAX)
        width = random.randint(max(CANVAS_MIN, max_w), CANVAS_MAX)
        if height * width < target_area:
            continue
        positions = _place_objects(objects, height, width)
        if positions is not None:
            return height, width, positions
    return None


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = patch != bg
    grid[top : top + patch.shape[0], left : left + patch.shape[1]][mask] = patch[mask]


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.choice(ALL_COLORS)
    non_bg = [c for c in ALL_COLORS if c != bg]
    frame_count = random.randint(1, 3)
    distractor_count = random.randint(0, 2)
    frame_colors = random.sample(non_bg, frame_count)
    sprite_colors = random.sample([c for c in non_bg if c not in frame_colors], frame_count + distractor_count)

    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    frames: list[dict[str, object]] = []
    sprites: list[dict[str, object]] = []

    for i in range(frame_count):
        outer_h = random.randint(6, 10)
        outer_w = random.randint(6, 10)
        built = _frame_patch(outer_h, outer_w, bg, frame_colors[i], used_shapes)
        if built is None:
            return None
        frame_obj, sprite_obj = built
        sprite_patch = sprite_obj["patch"]  # type: ignore[index]
        sprite_patch[sprite_patch == -1] = sprite_colors[i]
        frames.append(frame_obj)
        sprites.append(sprite_obj)

    for i in range(distractor_count):
        mask = _random_irregular_mask(5, 5, used_shapes)
        if mask is None:
            return None
        key = _shape_key_from_mask(mask)
        used_shapes.add(key)
        patch = np.full(mask.shape, bg, dtype=int)
        patch[mask] = sprite_colors[frame_count + i]
        sprites.append(
            {
                "kind": "sprite",
                "patch": patch,
                "matched": False,
                "shape_key": key,
            }
        )

    objects = frames + sprites
    random.shuffle(objects)
    sampled = _sample_canvas(objects, bg)
    if sampled is None:
        return None
    height, width, positions = sampled

    inp = np.full((height, width), bg, dtype=int)
    for obj, (top, left) in zip(objects, positions):
        _paste_patch(inp, obj["patch"], top, left, bg)  # type: ignore[arg-type]

    vals, counts = np.unique(inp, return_counts=True)
    bg_count = int(counts[np.where(vals == bg)[0][0]])
    other_max = max((int(c) for v, c in zip(vals, counts) if int(v) != bg), default=0)
    if bg_count <= other_max:
        return None

    out = inp.copy()
    for obj, (top, left) in zip(objects, positions):
        if obj["kind"] != "frame":
            continue
        patch = obj["patch"]  # type: ignore[index]
        hole_mask = obj["hole_mask"]  # type: ignore[index]
        frame_color = int(obj["frame_color"])
        interior = out[top + 1 : top + patch.shape[0] - 1, left + 1 : left + patch.shape[1] - 1]
        interior[hole_mask] = frame_color

    for obj, (top, left) in zip(objects, positions):
        if obj["kind"] != "sprite" or not obj["matched"]:
            continue
        patch = obj["patch"]  # type: ignore[index]
        mask = patch != bg
        out[top : top + patch.shape[0], left : left + patch.shape[1]][mask] = bg

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_case()
        if built is None:
            continue
        inp, _ = built
        return inp
    raise RuntimeError("failed to generate a valid ARC instance")
