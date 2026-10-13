from __future__ import annotations

from collections import Counter
import random
import sys
from pathlib import Path

import numpy as np


_ARC_ROOT = Path("/home/lixinyu/snap/new")
if str(_ARC_ROOT) not in sys.path:
    sys.path.append(str(_ARC_ROOT))

try:
    from solvers.arc_utils import crop_bbox, find_components, get_bg, paste, scale_sprite
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])

    def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = map(int, bbox)
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()

    def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
        k = int(factor)
        return np.kron(sprite, np.ones((k, k), dtype=sprite.dtype))

    def paste(
        dst: np.ndarray,
        patch: np.ndarray,
        r0: int,
        c0: int,
        *,
        transparent: int | None = None,
        clip: bool = False,
    ) -> np.ndarray:
        r0 = int(r0)
        c0 = int(c0)
        h, w = map(int, patch.shape)
        H, W = dst.shape
        rr0 = max(0, r0) if clip else r0
        cc0 = max(0, c0) if clip else c0
        rr1 = min(H, r0 + h) if clip else r0 + h
        cc1 = min(W, c0 + w) if clip else c0 + w
        if rr0 >= rr1 or cc0 >= cc1:
            return dst
        pr0 = rr0 - r0
        pc0 = cc0 - c0
        patch_view = patch[pr0 : pr0 + (rr1 - rr0), pc0 : pc0 + (cc1 - cc0)]
        dst_view = dst[rr0:rr1, cc0:cc1]
        if transparent is None:
            dst_view[:] = patch_view
        else:
            mask = patch_view != int(transparent)
            dst_view[mask] = patch_view[mask]
        return dst

    def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
        return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
    ) -> list[dict]:
        assert int(connectivity) == 4 and monochromatic
        g = np.asarray(grid, dtype=int)
        if bg is None:
            bg = get_bg(g)
        H, W = g.shape
        seen = np.zeros((H, W), dtype=bool)
        components: list[dict] = []
        for r in range(H):
            for c in range(W):
                color = int(g[r, c])
                if color == int(bg) or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                coords: list[tuple[int, int]] = []
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for nr, nc in _neighbors4(rr, cc):
                        if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == color:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                arr = np.asarray(coords, dtype=int)
                r0, c0 = arr.min(axis=0)
                r1, c1 = arr.max(axis=0)
                components.append(
                    {
                        "color": color,
                        "coords": arr,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "area": int(arr.shape[0]),
                    }
                )
        return components


BG = 0
COLOR_POOL = tuple(range(1, 10))
GRID_SIDE_RANGE = (22, 30)
FAMILY_COUNT_CHOICES = (3, 4, 4)
PLACEMENT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 500

GAP_INTERVALS = ((1, 3), (2, 4), (1, 4))
COMPLETE_INTERVALS = ((1, 2), (2, 3), (3, 4), (1, 3), (2, 4))


ShapeKey = tuple[tuple[int, ...], ...]


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_signature(mask: np.ndarray) -> ShapeKey:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _component_mask(grid: np.ndarray, component: dict) -> np.ndarray:
    return crop_bbox(grid, component["bbox"]) == int(component["color"])


def _normalize_scaled_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = map(int, mask.shape)
    for scale in range(min(h, w), 0, -1):
        if h % scale != 0 or w % scale != 0:
            continue
        blocks = mask.reshape(h // scale, scale, w // scale, scale)
        if not np.all(blocks == blocks[:, :1, :, :1]):
            continue
        base = blocks[:, 0, :, 0].astype(bool)
        rebuilt = scale_sprite(base.astype(int), scale).astype(bool)
        if np.array_equal(rebuilt, mask):
            return base, int(scale)
    raise AssertionError("every sprite must admit at least scale 1")


def _render_scaled_mask(mask: np.ndarray, color: int, scale: int, bg: int = BG) -> np.ndarray:
    scaled = scale_sprite(mask.astype(int), scale).astype(bool)
    patch = np.full(scaled.shape, int(bg), dtype=int)
    patch[scaled] = int(color)
    return patch


def _interval_len(interval: tuple[int, int]) -> int:
    return int(interval[1] - interval[0] + 1)


def _interval_scales(interval: tuple[int, int]) -> list[int]:
    return list(range(int(interval[0]), int(interval[1]) + 1))


def _scene_summary(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert len(components) >= 2, "expected multiple separated sprites"

    families: dict[tuple[int, ShapeKey], dict] = {}
    for component in components:
        mask = _component_mask(g, component)
        base_mask, scale = _normalize_scaled_mask(mask)
        key = (int(component["color"]), _mask_signature(base_mask))
        family = families.setdefault(
            key,
            {
                "color": int(component["color"]),
                "base_mask": base_mask.copy(),
                "scales": set(),
                "component_count": 0,
            },
        )
        assert np.array_equal(family["base_mask"], base_mask), "shape key must be stable"
        assert scale not in family["scales"], "duplicate scale within one family is outside this task"
        family["scales"].add(int(scale))
        family["component_count"] += 1

    culprits: list[tuple[tuple[int, ShapeKey], int]] = []
    for key, family in families.items():
        scales = sorted(int(v) for v in family["scales"])
        family["scales"] = scales
        family["interval"] = (scales[0], scales[-1])
        missing = sorted(set(range(scales[0], scales[-1] + 1)) - set(scales))
        assert len(missing) <= 1, "expected at most one internal gap per family"
        family["missing_scale"] = missing[0] if missing else None
        if missing:
            culprits.append((key, missing[0]))

    assert len(culprits) == 1, "expected exactly one family with an internal gap"
    culprit_key, missing_scale = culprits[0]
    culprit = families[culprit_key]
    expected = _render_scaled_mask(culprit["base_mask"], culprit["color"], missing_scale, bg)
    return {
        "bg": bg,
        "families": families,
        "culprit_key": culprit_key,
        "missing_scale": int(missing_scale),
        "expected": expected,
        "component_count": len(components),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the scale-equivalent sprite family with the one missing internal scale and render that missing upscale.

    Input structure:
    - The grid has a black background and several separated monochromatic 4-connected sprites.
    - Within each family, sprites share one color and one base shape, differing only by integer pixel scaling.
    - Exactly one family is missing one interior scale from an otherwise consecutive scale interval, while every other family is complete.

    Rule:
    - Crop each sprite to its bbox, normalize it to its scale-1 base mask, and group sprites by color plus normalized shape.
    - For every group, collect the observed integer scales and find the unique group whose scales skip exactly one internal value.
    - Output the minimal black canvas containing that family’s base shape scaled by the missing factor and painted in the family color.

    Key insight:
    - The answer depends on scale-equivalence classes rather than raw size, count, or position; only the family whose scale run has an internal gap determines the output.
    """
    return np.asarray(_scene_summary(np.asarray(grid, dtype=int))["expected"], dtype=int)


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        frontier = {
            nbr
            for r, c in cells
            for nbr in _neighbors4(r, c, height, width)
            if nbr not in cells
        }
        cells.add(random.choice(tuple(frontier)))
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _crop_mask(mask)


def _sample_base_mask() -> np.ndarray:
    for _ in range(200):
        if random.random() < 0.3:
            h = random.randint(3, 4)
            w = random.randint(3, 4)
            mask = np.ones((h, w), dtype=bool)
            mask[1:-1, 1:-1] = False
        else:
            h = random.randint(2, 4)
            w = random.randint(2, 4)
            max_area = min(h * w - 1, 8)
            area = random.randint(3, max_area)
            mask = _random_connected_mask(h, w, area)

        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, random.randrange(4))
        mask = _crop_mask(mask)

        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) < 3:
            continue
        if _normalize_scaled_mask(mask)[1] != 1:
            continue
        return mask.astype(bool)
    raise RuntimeError("failed to sample a primitive base mask")


def _sample_intervals(family_count: int) -> list[dict]:
    gap_interval = random.choice(GAP_INTERVALS)
    gap_missing = random.randint(gap_interval[0] + 1, gap_interval[1] - 1)
    gap_count = _interval_len(gap_interval) - 1
    same_count_choices = [
        interval
        for interval in COMPLETE_INTERVALS
        if interval != gap_interval and _interval_len(interval) == gap_count
    ]
    same_count_interval = random.choice(same_count_choices)

    remaining = [interval for interval in COMPLETE_INTERVALS if interval not in {gap_interval, same_count_interval}]
    extra_needed = family_count - 2
    extra_intervals = random.sample(remaining, extra_needed)

    families = [
        {
            "interval": gap_interval,
            "scales": [scale for scale in _interval_scales(gap_interval) if scale != gap_missing],
            "missing_scale": int(gap_missing),
        },
        {
            "interval": same_count_interval,
            "scales": _interval_scales(same_count_interval),
            "missing_scale": None,
        },
    ]
    for interval in extra_intervals:
        families.append(
            {
                "interval": interval,
                "scales": _interval_scales(interval),
                "missing_scale": None,
            }
        )
    random.shuffle(families)
    return families


def _sample_family_colors(family_count: int) -> list[int]:
    colors = random.sample(COLOR_POOL, family_count)
    if family_count >= 3 and random.random() < 0.4:
        src, dst = random.sample(range(family_count), 2)
        colors[dst] = colors[src]
    return [int(color) for color in colors]


def _build_family_specs() -> tuple[list[dict], dict]:
    family_count = random.choice(FAMILY_COUNT_CHOICES)
    interval_specs = _sample_intervals(family_count)
    colors = _sample_family_colors(family_count)
    used_keys: set[tuple[int, ShapeKey]] = set()

    families: list[dict] = []
    for spec, color in zip(interval_specs, colors):
        for _ in range(200):
            base_mask = _sample_base_mask()
            key = (int(color), _mask_signature(base_mask))
            if key in used_keys:
                continue
            used_keys.add(key)
            family = {
                "color": int(color),
                "base_mask": base_mask,
                "family_key": key,
                **spec,
            }
            families.append(family)
            break
        else:
            raise RuntimeError("failed to assign a unique family shape")

    gap_family = next(family for family in families if family["missing_scale"] is not None)
    return families, gap_family


def _build_sprite_list(families: list[dict]) -> list[dict]:
    sprites: list[dict] = []
    for family_id, family in enumerate(families):
        for scale in family["scales"]:
            patch = _render_scaled_mask(family["base_mask"], family["color"], scale)
            sprites.append(
                {
                    "family_id": family_id,
                    "scale": int(scale),
                    "patch": patch,
                    "shape": patch.shape,
                }
            )
    random.shuffle(sprites)
    return sprites


def _candidate_positions(reserved: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    H, W = reserved.shape
    positions: list[tuple[int, int]] = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            r0 = max(0, r - PLACEMENT_PADDING)
            c0 = max(0, c - PLACEMENT_PADDING)
            r1 = min(H, r + h + PLACEMENT_PADDING)
            c1 = min(W, c + w + PLACEMENT_PADDING)
            if not bool(reserved[r0:r1, c0:c1].any()):
                positions.append((r, c))
    random.shuffle(positions)
    return positions


def _place_sprites(sprites: list[dict], height: int, width: int) -> np.ndarray | None:
    reserved = np.zeros((height, width), dtype=bool)
    locations: list[tuple[int, int] | None] = [None] * len(sprites)
    order = list(range(len(sprites)))
    order.sort(key=lambda idx: sprites[idx]["shape"][0] * sprites[idx]["shape"][1], reverse=True)

    def backtrack(depth: int) -> bool:
        if depth == len(order):
            return True
        idx = order[depth]
        sprite = sprites[idx]
        h, w = map(int, sprite["shape"])
        for r, c in _candidate_positions(reserved, sprite["shape"]):
            reserved[r : r + h, c : c + w] = True
            locations[idx] = (r, c)
            if backtrack(depth + 1):
                return True
            locations[idx] = None
            reserved[r : r + h, c : c + w] = False
        return False

    if not backtrack(0):
        return None

    grid = np.full((height, width), BG, dtype=int)
    for sprite, loc in zip(sprites, locations):
        assert loc is not None
        paste(grid, sprite["patch"], loc[0], loc[1], transparent=BG)
    return grid


def _validate_case(grid: np.ndarray, families: list[dict], gap_family: dict) -> bool:
    expected = _render_scaled_mask(gap_family["base_mask"], gap_family["color"], gap_family["missing_scale"])
    try:
        summary = _scene_summary(grid)
        actual = main(grid)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if not np.array_equal(summary["expected"], expected):
        return False
    if summary["culprit_key"] != gap_family["family_key"]:
        return False
    if summary["missing_scale"] != gap_family["missing_scale"]:
        return False
    if summary["component_count"] != sum(len(family["scales"]) for family in families):
        return False

    family_info = summary["families"]
    gap_present_count = len(gap_family["scales"])
    complete_same_count = [
        info
        for key, info in family_info.items()
        if key != gap_family["family_key"] and info["missing_scale"] is None and len(info["scales"]) == gap_present_count
    ]
    if not complete_same_count:
        return False

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if not any(len(info["scales"]) >= 3 for info in family_info.values()):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    families, gap_family = _build_family_specs()
    sprites = _build_sprite_list(families)

    max_h = max(int(sprite["shape"][0]) for sprite in sprites)
    max_w = max(int(sprite["shape"][1]) for sprite in sprites)
    height = random.randint(max(GRID_SIDE_RANGE[0], max_h + 2), GRID_SIDE_RANGE[1])
    width = random.randint(max(GRID_SIDE_RANGE[0], max_w + 2), GRID_SIDE_RANGE[1])

    padded_area = sum((sprite["shape"][0] + 1) * (sprite["shape"][1] + 1) for sprite in sprites)
    if padded_area > int(height * width * 0.6):
        return None

    grid = _place_sprites(sprites, height, width)
    if grid is None:
        return None
    if _validate_case(grid, families, gap_family):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid internal-gap scale-family puzzle")
