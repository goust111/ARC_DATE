from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, paste


BG = Color.BLACK
GEN_ATTEMPTS = 500
SPRITE_ATTEMPTS = 120
SIZE_OPTIONS = (3, 5, 7)


def _vector_category(vector: tuple[int, int]) -> str:
    dr, dc = map(int, vector)
    if dr == 0 and dc != 0:
        return "horizontal"
    if dc == 0 and dr != 0:
        return "vertical"
    if dr != 0 and abs(dr) == abs(dc):
        return "diagonal"
    raise AssertionError(f"invalid instruction vector {vector}")


def _max_jumps_for_bbox(
    bbox: tuple[int, int, int, int],
    vector: tuple[int, int],
    shape: tuple[int, int],
) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    dr, dc = map(int, vector)
    height, width = map(int, shape)
    limits: list[int] = []

    if dr > 0:
        limits.append((height - 1 - r1) // dr)
    elif dr < 0:
        limits.append(r0 // (-dr))

    if dc > 0:
        limits.append((width - 1 - c1) // dc)
    elif dc < 0:
        limits.append(c0 // (-dc))

    assert limits and any(step != 0 for step in (dr, dc))
    return int(min(limits))


def _shift_bbox(
    bbox: tuple[int, int, int, int],
    jumps: int,
    vector: tuple[int, int],
) -> tuple[int, int, int, int]:
    dr, dc = map(int, vector)
    delta_r = int(jumps) * dr
    delta_c = int(jumps) * dc
    r0, c0, r1, c1 = map(int, bbox)
    return r0 + delta_r, c0 + delta_c, r1 + delta_r, c1 + delta_c


def _dilate8(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    padded = np.pad(mask.astype(bool), 1)
    out = np.zeros((height, width), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr : dr + height, dc : dc + width]
    return out


def _path_mask(
    patch: np.ndarray,
    top: int,
    left: int,
    vector: tuple[int, int],
    jumps: int,
    shape: tuple[int, int],
) -> np.ndarray:
    dr, dc = map(int, vector)
    mask_coords = np.argwhere(patch != BG)
    out = np.zeros(shape, dtype=bool)
    for step in range(int(jumps) + 1):
        rows = mask_coords[:, 0] + int(top) + step * dr
        cols = mask_coords[:, 1] + int(left) + step * dc
        out[rows, cols] = True
    return out


def _has_visible_gap(
    bbox: tuple[int, int, int, int],
    vector: tuple[int, int],
    shape: tuple[int, int],
    jumps: int,
) -> bool:
    final_bbox = _shift_bbox(bbox, jumps, vector)
    r0, c0, r1, c1 = map(int, final_bbox)
    dr, dc = map(int, vector)
    height, width = map(int, shape)
    checks: list[bool] = []

    if dr > 0:
        gap = height - 1 - r1
        checks.append(0 < gap < dr)
    elif dr < 0:
        gap = r0
        checks.append(0 < gap < -dr)

    if dc > 0:
        gap = width - 1 - c1
        checks.append(0 < gap < dc)
    elif dc < 0:
        gap = c0
        checks.append(0 < gap < -dc)

    return any(checks)


def _analyze_component(grid: np.ndarray, component: dict) -> dict:
    patch = crop_bbox(grid, component["bbox"])
    height, width = patch.shape
    assert height % 2 == 1 and width % 2 == 1, "sprite bbox must be odd-sized"

    colors = [int(color) for color in np.unique(patch) if int(color) != BG]
    assert len(colors) == 2, "each sprite must use one body color and one instruction color"

    counts = {color: int(np.sum(patch == color)) for color in colors}
    instruction_colors = [color for color, count in counts.items() if count == 1]
    assert len(instruction_colors) == 1, "expected exactly one singleton instruction color"
    instruction_color = int(instruction_colors[0])
    body_color = int([color for color in colors if color != instruction_color][0])
    assert counts[body_color] >= 2, "body must occupy multiple cells"

    instruction_rel = tuple(map(int, np.argwhere(patch == instruction_color)[0]))
    center = (height // 2, width // 2)
    vector = (
        int(instruction_rel[0] - center[0]),
        int(instruction_rel[1] - center[1]),
    )
    _vector_category(vector)

    bbox = tuple(map(int, component["bbox"]))
    jumps = _max_jumps_for_bbox(bbox, vector, grid.shape)
    assert jumps >= 1, "every generated sprite should move at least once"

    return {
        "bbox": bbox,
        "patch": patch,
        "vector": vector,
        "jumps": jumps,
        "body_color": body_color,
        "instruction_color": instruction_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Jump each sprite repeatedly by the vector from its center to its lone instruction pixel.

    Input structure:
    - A black background contains several separated multicolor sprites.
    - Each sprite is monochrome except for one singleton instruction pixel of another color.
    - Every sprite's odd-sized bounding box gives an integer center, and the instruction pixel lies on the same row, column, or 45-degree diagonal as that center.

    Rule:
    - For each sprite, crop its bounding box and find the offset vector from the bbox center to the singleton instruction pixel.
    - Translate the entire sprite by that fixed vector over and over.
    - Stop just before the next translation would make any sprite cell leave the grid, and place every sprite at that final location on a black canvas.

    Key insight:
    - The odd one-pixel color change is not decoration: its offset from the bbox center simultaneously encodes jump direction and jump length, so sprites stop at step-aligned positions rather than sliding all the way to a wall.
    """
    g = np.asarray(grid, dtype=int)
    components = find_components(
        g,
        bg=BG,
        connectivity=4,
        monochromatic=False,
    )
    out = np.full_like(g, BG)

    for component in components:
        info = _analyze_component(g, component)
        bbox = info["bbox"]
        patch = info["patch"]
        jumps = int(info["jumps"])
        dr, dc = map(int, info["vector"])
        dest_top = bbox[0] + jumps * dr
        dest_left = bbox[1] + jumps * dc
        target = out[dest_top : dest_top + patch.shape[0], dest_left : dest_left + patch.shape[1]]
        assert np.all(target[patch != BG] == BG), "sprite targets must not overlap"
        paste(out, patch, dest_top, dest_left, transparent=BG)

    return out


def _instruction_candidates(
    mask: np.ndarray,
    category: str,
    *,
    min_step: int,
) -> list[tuple[int, int, int, int, int]]:
    center_r = mask.shape[0] // 2
    center_c = mask.shape[1] // 2
    candidates: list[tuple[int, int, int, int, int]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        dr = rr - center_r
        dc = cc - center_c
        if dr == 0 and dc == 0:
            continue
        if category == "horizontal" and dr != 0:
            continue
        if category == "vertical" and dc != 0:
            continue
        if category == "diagonal" and abs(dr) != abs(dc):
            continue
        step = max(abs(dr), abs(dc))
        if step < int(min_step):
            continue
        candidates.append((rr, cc, dr, dc, step))
    return candidates


def _grow_body_mask(height: int, width: int) -> np.ndarray:
    area = int(height) * int(width)
    lo = max(4, int(np.ceil(0.42 * area)))
    hi = max(lo, int(np.ceil(0.78 * area)))
    target = random.randint(lo, hi)

    center = (int(height) // 2, int(width) // 2)
    mask = np.zeros((height, width), dtype=bool)
    mask[center] = True
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < height and 0 <= cc < width and not mask[rr, cc]:
                frontier.add((rr, cc))

    add_frontier(*center)

    def touches_all_sides() -> bool:
        return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())

    while frontier and (int(mask.sum()) < target or not touches_all_sides()):
        cells = list(frontier)
        weights = []
        for r, c in cells:
            edge_bonus = int(r == 0 or r == height - 1 or c == 0 or c == width - 1)
            neighbor_count = sum(
                0 <= r + dr < height and 0 <= c + dc < width and mask[r + dr, c + dc]
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
            center_bias = 1 / (1 + abs(r - center[0]) + abs(c - center[1]))
            weights.append((neighbor_count + 1) ** 2 + edge_bonus + center_bias)
        chosen_r, chosen_c = random.choices(cells, weights=weights, k=1)[0]
        frontier.remove((chosen_r, chosen_c))
        mask[chosen_r, chosen_c] = True
        add_frontier(chosen_r, chosen_c)

    if not touches_all_sides():
        return _grow_body_mask(height, width)
    return mask


def _make_sprite_patch(
    category: str,
    body_color: int,
    instruction_color: int,
    *,
    min_step: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    size_pool = [side for side in SIZE_OPTIONS if side >= 2 * int(min_step) + 1]
    assert size_pool

    for _ in range(SPRITE_ATTEMPTS):
        if category == "diagonal":
            side = random.choice(size_pool)
            height = width = side
        else:
            if random.random() < 0.35:
                side = random.choice(size_pool)
                height = width = side
            else:
                height = random.choice(size_pool)
                width = random.choice(SIZE_OPTIONS)

        patch = np.where(_grow_body_mask(height, width), int(body_color), BG)
        mask = patch != BG
        candidates = _instruction_candidates(mask, category, min_step=min_step)
        if not candidates:
            continue

        weights = [step * step for _r, _c, _dr, _dc, step in candidates]
        mark_r, mark_c, dr, dc, _step = random.choices(candidates, weights=weights, k=1)[0]
        sprite = patch.copy()
        sprite[mark_r, mark_c] = int(instruction_color)
        return sprite, (int(dr), int(dc))

    raise RuntimeError(f"failed to generate a {category} sprite")


def _placement_candidates(
    shape: tuple[int, int],
    patch: np.ndarray,
    vector: tuple[int, int],
    jump_options: tuple[int, ...],
    reserved_dilated: np.ndarray,
    *,
    require_visible_gap: bool,
) -> list[dict]:
    height, width = map(int, shape)
    patch_h, patch_w = patch.shape
    candidates: list[dict] = []
    mask_coords = np.argwhere(patch != BG)

    for top in range(height - patch_h + 1):
        for left in range(width - patch_w + 1):
            bbox = (top, left, top + patch_h - 1, left + patch_w - 1)
            jumps = _max_jumps_for_bbox(bbox, vector, shape)
            if jumps not in jump_options:
                continue
            if require_visible_gap and not _has_visible_gap(bbox, vector, shape, jumps):
                continue

            path = np.zeros(shape, dtype=bool)
            ok = True
            for step in range(jumps + 1):
                rows = mask_coords[:, 0] + top + step * vector[0]
                cols = mask_coords[:, 1] + left + step * vector[1]
                if np.any(reserved_dilated[rows, cols]):
                    ok = False
                    break
                path[rows, cols] = True
            if not ok:
                continue

            candidates.append(
                {
                    "top": int(top),
                    "left": int(left),
                    "jumps": int(jumps),
                    "path": path,
                }
            )

    return candidates


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(18, 30)
    width = random.randint(18, 30)
    n_sprites = random.choice([3, 4])

    categories = ["horizontal", "vertical", "diagonal"]
    if n_sprites == 4:
        categories.append(random.choice(categories))
    random.shuffle(categories)

    all_colors = random.sample([int(color) for color in Color.NOT_BLACK], 2 * n_sprites)
    specs = []
    visible_gap_index = random.randrange(n_sprites)
    for idx, category in enumerate(categories):
        specs.append(
            {
                "category": category,
                "body_color": all_colors[2 * idx],
                "instruction_color": all_colors[2 * idx + 1],
                "min_step": 2 if idx == visible_gap_index else 1,
                "jump_options": (2, 3, 4) if idx == visible_gap_index else (1, 2, 3),
                "require_visible_gap": idx == visible_gap_index,
            }
        )

    specs.sort(
        key=lambda spec: (
            not spec["require_visible_gap"],
            -max(spec["jump_options"]),
            spec["category"],
        )
    )

    grid = np.full((height, width), BG, dtype=int)
    expected = np.full((height, width), BG, dtype=int)
    reserved_dilated = np.zeros((height, width), dtype=bool)

    for spec in specs:
        placed = False
        for _ in range(SPRITE_ATTEMPTS):
            patch, vector = _make_sprite_patch(
                spec["category"],
                spec["body_color"],
                spec["instruction_color"],
                min_step=spec["min_step"],
            )
            candidates = _placement_candidates(
                (height, width),
                patch,
                vector,
                spec["jump_options"],
                reserved_dilated,
                require_visible_gap=spec["require_visible_gap"],
            )
            if not candidates:
                continue

            choice = random.choice(candidates)
            top = int(choice["top"])
            left = int(choice["left"])
            jumps = int(choice["jumps"])
            path = choice["path"]
            dest_top = top + jumps * vector[0]
            dest_left = left + jumps * vector[1]

            paste(grid, patch, top, left, transparent=BG)
            target = expected[dest_top : dest_top + patch.shape[0], dest_left : dest_left + patch.shape[1]]
            if np.any(target[patch != BG] != BG):
                return None
            paste(expected, patch, dest_top, dest_left, transparent=BG)
            reserved_dilated |= _dilate8(path)
            placed = True
            break

        if not placed:
            return None

    if not _scene_is_semantic(grid, expected):
        return None
    return grid, expected


def _scene_is_semantic(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        components = find_components(
            inp,
            bg=BG,
            connectivity=4,
            monochromatic=False,
        )
        if len(components) < 3:
            return False

        kinds: set[str] = set()
        has_visible_gap = False
        has_repeated_jump = False
        rebuilt = np.full_like(inp, BG)
        reserved_dilated = np.zeros(inp.shape, dtype=bool)

        for component in components:
            info = _analyze_component(inp, component)
            bbox = info["bbox"]
            patch = info["patch"]
            vector = info["vector"]
            jumps = int(info["jumps"])
            kinds.add(_vector_category(vector))
            has_visible_gap |= _has_visible_gap(bbox, vector, inp.shape, jumps)
            has_repeated_jump |= jumps >= 2

            path = _path_mask(patch, bbox[0], bbox[1], vector, jumps, inp.shape)
            if np.any(path & reserved_dilated):
                return False
            reserved_dilated |= _dilate8(path)

            dest_top = bbox[0] + jumps * vector[0]
            dest_left = bbox[1] + jumps * vector[1]
            target = rebuilt[dest_top : dest_top + patch.shape[0], dest_left : dest_left + patch.shape[1]]
            if np.any(target[patch != BG] != BG):
                return False
            paste(rebuilt, patch, dest_top, dest_left, transparent=BG)

        if kinds != {"horizontal", "vertical", "diagonal"}:
            return False
        if not has_visible_gap or not has_repeated_jump:
            return False
        if not np.array_equal(rebuilt, expected):
            return False
        if not np.array_equal(main(inp), expected):
            return False
    except (AssertionError, RuntimeError, ValueError):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_scene()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid vector-jump puzzle input")
