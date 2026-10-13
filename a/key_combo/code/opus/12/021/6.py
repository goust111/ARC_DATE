from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_RETRIES = 400
GROUP_COUNT_OPTIONS = {
    4: [(1, 3)],
    5: [(1, 4), (1, 2, 2)],
    6: [(1, 5), (1, 2, 3)],
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_from_coords(coords: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _touches_crop_border(box: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = box
    h, w = shape
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _paint_mask(dst: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    view[mask] = int(color)


def _rectangles_with_gap_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _extract_cavities(template: np.ndarray) -> list[dict]:
    cavities: list[dict] = []
    for coords in _components(template == BLACK):
        box = _bbox(coords)
        if _touches_crop_border(box, template.shape):
            continue
        mask = _mask_from_coords(coords, box)
        assert bool(mask.all()), "expected every internal black component to be rectangular"
        r0, c0, r1, c1 = box
        cavities.append(
            {
                "bbox": box,
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
                "anchor": (int(r0), int(c0)),
            }
        )
    cavities.sort(key=lambda cav: cav["anchor"])
    return cavities


def _extract_sprites(
    grid: np.ndarray, template_box: tuple[int, int, int, int]
) -> list[dict]:
    r0, c0, r1, c1 = template_box
    masked = grid.copy()
    masked[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    sprites: list[dict] = []
    for coords in _components(masked == YELLOW):
        box = _bbox(coords)
        mask = _mask_from_coords(coords, box)
        br0, bc0, br1, bc1 = box
        sprites.append(
            {
                "bbox": box,
                "h": int(br1 - br0 + 1),
                "w": int(bc1 - bc0 + 1),
                "mask": mask,
                "shape_key": _shape_key(mask),
                "anchor": (int(br0), int(bc0)),
            }
        )
    sprites.sort(key=lambda sprite: sprite["anchor"])
    return sprites


def _solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    template_coords = np.argwhere(g == GREY)
    assert len(template_coords) > 0, "expected a grey template frame"
    template_box = _bbox(template_coords)
    template = _crop(g, template_box)

    cavities = _extract_cavities(template)
    assert 4 <= len(cavities) <= 6, "expected 4-6 cavities"

    sprites = _extract_sprites(g, template_box)
    assert len(sprites) == len(cavities), "sprite count must equal cavity count"

    cavity_dims = Counter((cav["h"], cav["w"]) for cav in cavities)
    sprite_dims = Counter((sp["h"], sp["w"]) for sp in sprites)
    assert cavity_dims == sprite_dims, "cavity dimensions must match sprite dimensions"

    same_size_groups: dict[tuple[int, int], list[dict]] = {}
    for sprite in sprites:
        same_size_groups.setdefault((sprite["h"], sprite["w"]), []).append(sprite)
    for group in same_size_groups.values():
        if len(group) > 1:
            keys = {sprite["shape_key"] for sprite in group}
            assert len(keys) == 1, "same-size sprites must be identical for unique embedding"

    shape_counts = Counter(sprite["shape_key"] for sprite in sprites)
    assert sum(count == 1 for count in shape_counts.values()) == 1, "expected exactly one singleton shape"
    assert any(count >= 2 for count in shape_counts.values()), "expected repeated shapes"

    return {
        "grid": g,
        "template_box": template_box,
        "template": template,
        "cavities": cavities,
        "sprites": sprites,
        "shape_counts": shape_counts,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Embed the outside sprites into the template cavities, then recolor them by shape frequency.

    Input structure:
    - The background is black, with one grey rectangular frame/template containing 4-6 black rectangular cavities.
    - Outside that template are 4-connected yellow sprites whose bounding-box dimensions match the cavity dimensions.
    - Cavities may repeat a size, but repeated same-size sprites are exact copies, so the size matching remains unique in effect.

    Rule:
    - Find the grey template and crop to its bounding box.
    - Find every internal rectangular black cavity and every yellow outside sprite, then match them by exact bounding-box size.
    - Compare sprites by their normalized binary shape; the unique shape is recolored red and every repeated shape is recolored blue after being pasted into its cavity.
    - Keep the grey frame unchanged, keep unused cavity cells black, and discard the outside sprites by returning only the template crop.

    Key insight:
    - The output needs both operations: size-based embedding determines where each sprite goes, while the final color depends on the global frequency of the normalized sprite shape, not on cavity position.
    """
    details = _solve_details(grid)
    out = details["template"].copy()

    cavity_groups: dict[tuple[int, int], list[dict]] = {}
    sprite_groups: dict[tuple[int, int], list[dict]] = {}
    for cavity in details["cavities"]:
        cavity_groups.setdefault((cavity["h"], cavity["w"]), []).append(cavity)
    for sprite in details["sprites"]:
        sprite_groups.setdefault((sprite["h"], sprite["w"]), []).append(sprite)

    for dims, cavities in cavity_groups.items():
        sprites = sprite_groups[dims]
        cavities = sorted(cavities, key=lambda cav: cav["anchor"])
        sprites = sorted(sprites, key=lambda sp: sp["anchor"])
        for cavity, sprite in zip(cavities, sprites):
            color = RED if details["shape_counts"][sprite["shape_key"]] == 1 else BLUE
            rr0, cc0, _rr1, _cc1 = cavity["bbox"]
            _paint_mask(out, sprite["mask"], rr0, cc0, color)

    return out


def _random_connected_shape(h: int, w: int) -> np.ndarray | None:
    max_area = min(8, h * w)
    if max_area < 3:
        return None

    for _ in range(120):
        target = random.randint(3, max_area)
        if h * w > 3 and target == h * w:
            target -= 1
        if target < 3:
            target = 3

        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True

        while int(mask.sum()) < target:
            frontier: list[tuple[int, int]] = []
            seen = set()
            for r, c in np.argwhere(mask):
                for dr, dc in DIR4:
                    nr = int(r) + dr
                    nc = int(c) + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            mask[nr, nc] = True

        if int(mask.sum()) != target:
            continue
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox(coords)
        if (r1 - r0 + 1, c1 - c0 + 1) != (h, w):
            continue
        if h > 1 and w > 1 and bool(mask.all()):
            continue
        if h >= 3 and all(mask[row].sum() == 0 for row in range(h)):
            continue
        return mask
    return None


def _build_template_case() -> dict | None:
    orientation = random.choice(["horizontal", "vertical"])
    cavity_count = random.randint(4, 6)
    count_pattern = list(random.choice(GROUP_COUNT_OPTIONS[cavity_count]))
    group_total = len(count_pattern)

    if orientation == "horizontal":
        cell_h = random.randint(2, 4)
        dims = random.sample([2, 3, 4], k=group_total)
        cavity_order: list[int] = []
        for gid, count in enumerate(count_pattern):
            cavity_order.extend([gid] * count)
        random.shuffle(cavity_order)

        widths = {gid: dims[gid] for gid in range(group_total)}
        template_h = cell_h + 2
        template_w = sum(widths[gid] for gid in cavity_order) + len(cavity_order) + 1
        if template_w > 27:
            return None

        crop = np.full((template_h, template_w), GREY, dtype=int)
        cavities: list[dict] = []
        c = 1
        for gid in cavity_order:
            w = widths[gid]
            crop[1 : 1 + cell_h, c : c + w] = BLACK
            cavities.append({"bbox": (1, c, cell_h, c + w - 1), "gid": gid, "dims": (cell_h, w)})
            c += w + 1
    else:
        cell_w = random.randint(2, 4)
        dims = random.sample([2, 3, 4], k=group_total)
        cavity_order = []
        for gid, count in enumerate(count_pattern):
            cavity_order.extend([gid] * count)
        random.shuffle(cavity_order)

        heights = {gid: dims[gid] for gid in range(group_total)}
        template_w = cell_w + 2
        template_h = sum(heights[gid] for gid in cavity_order) + len(cavity_order) + 1
        if template_h > 27:
            return None

        crop = np.full((template_h, template_w), GREY, dtype=int)
        cavities = []
        r = 1
        for gid in cavity_order:
            h = heights[gid]
            crop[r : r + h, 1 : 1 + cell_w] = BLACK
            cavities.append({"bbox": (r, 1, r + h - 1, cell_w), "gid": gid, "dims": (h, cell_w)})
            r += h + 1

    shapes: dict[int, np.ndarray] = {}
    for gid in range(group_total):
        h, w = cavities[cavity_order.index(gid)]["dims"]
        shape = _random_connected_shape(h, w)
        if shape is None:
            return None
        shapes[gid] = shape

    expected = crop.copy()
    for cavity in cavities:
        gid = cavity["gid"]
        rr0, cc0, _rr1, _cc1 = cavity["bbox"]
        color = RED if count_pattern[gid] == 1 else BLUE
        _paint_mask(expected, shapes[gid], rr0, cc0, color)

    return {
        "orientation": orientation,
        "crop": crop,
        "expected": expected,
        "cavities": cavities,
        "shapes": shapes,
        "counts": count_pattern,
    }


def _choose_canvas(template_h: int, template_w: int, orientation: str) -> tuple[int, int] | None:
    if orientation == "horizontal":
        h = template_h + random.randint(6, 10)
        w = template_w + random.randint(2, 4)
    else:
        h = template_h + random.randint(2, 4)
        w = template_w + random.randint(6, 10)
    if h > 30 or w > 30:
        return None
    return h, w


def _scatter_sprites(
    grid: np.ndarray,
    patches: list[np.ndarray],
    template_box: tuple[int, int, int, int],
) -> bool:
    h, w = grid.shape
    placed_boxes: list[tuple[int, int, int, int]] = []
    indices = list(range(len(patches)))
    random.shuffle(indices)
    indices.sort(key=lambda idx: int((patches[idx] == YELLOW).sum()), reverse=True)

    for idx in indices:
        patch = patches[idx]
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(h - ph + 1):
            for c0 in range(w - pw + 1):
                box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if _rectangles_with_gap_overlap(box, template_box, 1):
                    continue
                if any(_rectangles_with_gap_overlap(box, other, 1) for other in placed_boxes):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        grid[r0 : r0 + ph, c0 : c0 + pw][patch == YELLOW] = YELLOW
        placed_boxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
    return True


def _validate_generated(inp: np.ndarray, expected: np.ndarray) -> None:
    details = _solve_details(inp)
    out = main(inp)
    assert np.array_equal(out, expected)
    assert np.array_equal(out, details["template"]) is False
    assert Counter(out.ravel())[GREY] > 0
    assert Counter(out.ravel())[RED] > 0
    assert Counter(out.ravel())[BLUE] > 0


def _attempt_generate_input() -> np.ndarray | None:
    case = _build_template_case()
    if case is None:
        return None

    crop = case["crop"]
    expected = case["expected"]
    template_h, template_w = crop.shape
    canvas_shape = _choose_canvas(template_h, template_w, case["orientation"])
    if canvas_shape is None:
        return None
    canvas_h, canvas_w = canvas_shape

    if canvas_h - template_h < 2 or canvas_w - template_w < 2:
        return None

    grid = np.zeros((canvas_h, canvas_w), dtype=int)
    template_r0 = random.randint(1, canvas_h - template_h - 1)
    template_c0 = random.randint(1, canvas_w - template_w - 1)
    grid[template_r0 : template_r0 + template_h, template_c0 : template_c0 + template_w] = crop
    template_box = (
        template_r0,
        template_c0,
        template_r0 + template_h - 1,
        template_c0 + template_w - 1,
    )

    patches: list[np.ndarray] = []
    for cavity in case["cavities"]:
        gid = cavity["gid"]
        mask = case["shapes"][gid]
        patch = np.where(mask, YELLOW, BLACK).astype(int)
        patches.append(patch)

    if not _scatter_sprites(grid, patches, template_box):
        return None

    try:
        _validate_generated(grid, expected)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _attempt_generate_input()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input for puzzle 6")
