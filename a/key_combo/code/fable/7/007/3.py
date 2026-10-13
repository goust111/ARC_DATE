from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("arc_utils_shared", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

bbox_coords = _ARC_UTILS.bbox_coords
colors_present = _ARC_UTILS.colors_present
components_mask = _ARC_UTILS.components_mask
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
paste = _ARC_UTILS.paste


GRID_MIN = 22
GRID_MAX = 30
TEMPLATE_H_RANGE = (11, 18)
TEMPLATE_W_RANGE = (11, 18)
CAVITY_COUNT_RANGE = (3, 5)
REPEATED_GROUP_SIZE = (2, 3)
SIZE_OPTIONS = (
    (3, 3),
    (3, 4),
    (4, 3),
    (4, 4),
    (3, 5),
    (5, 3),
    (4, 5),
    (5, 4),
)
GEN_RETRIES = 400


def _border_background(grid: np.ndarray) -> int:
    border = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    vals, counts = np.unique(border, return_counts=True)
    return int(vals[np.argmax(counts)])


def _largest_template_component(grid: np.ndarray, bg: int) -> tuple[int, np.ndarray]:
    best: tuple[int, int, np.ndarray] | None = None
    ambiguous = False
    for color in colors_present(grid, exclude={bg}):
        for coords in components_mask(grid == int(color), connectivity=4):
            area = int(coords.shape[0])
            if best is None or area > best[0]:
                best = (area, int(color), coords)
                ambiguous = False
            elif area == best[0]:
                ambiguous = True
    assert best is not None and not ambiguous
    return int(best[1]), best[2]


def _mask_connected(mask: np.ndarray) -> bool:
    comps = components_mask(mask.astype(bool), connectivity=4)
    return len(comps) == 1


def _mask_spans_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _mask_signature(mask: np.ndarray) -> bytes:
    return mask.astype(np.uint8).tobytes() + str(mask.shape).encode()


def _rect_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return not (
        ar + ah - 1 + gap < br
        or br + bh - 1 + gap < ar
        or ac + aw - 1 + gap < bc
        or bc + bw - 1 + gap < ac
    )


def _random_spanning_sprite(h: int, w: int) -> np.ndarray | None:
    if h < 3 or w < 3:
        return None
    mask = np.ones((h, w), dtype=bool)
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(interior)
    max_removed = len(interior) - 1 if len(interior) > 1 else 1
    remove_count = random.randint(1, max_removed)
    for r, c in interior[:remove_count]:
        mask[r, c] = False
    if not (~mask).any():
        return None
    return mask


def _choose_cavity_sizes() -> list[tuple[int, int]]:
    n = random.randint(*CAVITY_COUNT_RANGE)
    repeat_candidates = [size for size in SIZE_OPTIONS if (size[0] - 2) * (size[1] - 2) >= 2]
    repeat_size = random.choice(repeat_candidates)
    repeat_count = random.randint(*REPEATED_GROUP_SIZE)
    repeat_count = min(repeat_count, n)
    sizes = [repeat_size] * repeat_count
    pool = [size for size in SIZE_OPTIONS if size != repeat_size]
    random.shuffle(pool)
    sizes.extend(pool[: n - repeat_count])
    random.shuffle(sizes)
    return sizes


def _place_cavities(template_h: int, template_w: int, sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    blocked = np.zeros((template_h, template_w), dtype=bool)
    placements: list[tuple[int, int, int, int]] = []
    for h, w in sorted(sizes, key=lambda x: x[0] * x[1], reverse=True):
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, template_h - h):
            for c0 in range(1, template_w - w):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(template_h, r0 + h + 1)
                cc1 = min(template_w, c0 + w + 1)
                if not blocked[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements.append((r0, c0, h, w))
        blocked[max(0, r0 - 1) : min(template_h, r0 + h + 1), max(0, c0 - 1) : min(template_w, c0 + w + 1)] = True
    random.shuffle(placements)
    return placements


def _build_template_instance() -> tuple[np.ndarray, list[dict], int, int] | None:
    template_h = random.randint(*TEMPLATE_H_RANGE)
    template_w = random.randint(*TEMPLATE_W_RANGE)
    sizes = _choose_cavity_sizes()
    placements = _place_cavities(template_h, template_w, sizes)
    if placements is None:
        return None

    bg = 0
    base = random.choice(list(range(1, 10)))
    sprite_colors = [c for c in range(1, 10) if c != base]
    if len(sprite_colors) < len(placements):
        return None
    random.shuffle(sprite_colors)

    template = np.full((template_h, template_w), base, dtype=int)
    cavities: list[dict] = []
    used_masks_by_size: dict[tuple[int, int], set[bytes]] = {}

    for idx, (r0, c0, h, w) in enumerate(placements):
        mask = None
        for _ in range(80):
            candidate = _random_spanning_sprite(h, w)
            if candidate is None:
                continue
            sig = _mask_signature(candidate)
            bucket = used_masks_by_size.setdefault((h, w), set())
            if sig in bucket:
                continue
            bucket.add(sig)
            mask = candidate
            break
        if mask is None:
            return None

        residue = ~mask
        rect = np.full((h, w), bg, dtype=int)
        rect[residue] = base
        template[r0 : r0 + h, c0 : c0 + w] = rect
        cavities.append(
            {
                "bbox": (r0, c0, h, w),
                "sprite_mask": mask,
                "residue_mask": residue,
                "color": int(sprite_colors[idx]),
            }
        )

    size_hist: dict[tuple[int, int], int] = {}
    for cavity in cavities:
        size = cavity["bbox"][2], cavity["bbox"][3]
        size_hist[size] = size_hist.get(size, 0) + 1
    if max(size_hist.values()) < 2:
        return None

    return template, cavities, bg, base


def _choose_canvas(template_h: int, template_w: int) -> tuple[int, int, int, int]:
    H = random.randint(max(GRID_MIN, template_h + 7), GRID_MAX)
    W = random.randint(max(GRID_MIN, template_w + 7), GRID_MAX)
    r0 = random.randint(2, H - template_h - 2)
    c0 = random.randint(2, W - template_w - 2)
    return H, W, r0, c0


def _place_sources(
    grid: np.ndarray,
    template_rect: tuple[int, int, int, int],
    cavities: list[dict],
    bg: int,
) -> bool:
    placed_rects: list[tuple[int, int, int, int]] = [template_rect]
    H, W = grid.shape

    items = cavities[:]
    random.shuffle(items)
    for cavity in items:
        sprite = np.full(cavity["sprite_mask"].shape, bg, dtype=int)
        sprite[cavity["sprite_mask"]] = int(cavity["color"])
        h, w = sprite.shape

        candidates: list[tuple[int, int]] = []
        for r0 in range(1, H - h):
            for c0 in range(1, W - w):
                rect = (r0, c0, h, w)
                if any(_rect_intersects(rect, other, gap=1) for other in placed_rects):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return False

        r0, c0 = random.choice(candidates)
        paste(grid, sprite, r0, c0, transparent=bg)
        placed_rects.append((r0, c0, h, w))
    return True


def _expected_output(template: np.ndarray, cavities: list[dict], base: int) -> np.ndarray:
    out = template.copy()
    for cavity in cavities:
        r0, c0, h, w = cavity["bbox"]
        patch = out[r0 : r0 + h, c0 : c0 + w]
        assert np.all((patch == 0) | (patch == base))
        patch[cavity["sprite_mask"]] = int(cavity["color"])
    return out


def _extract_cavities(crop: np.ndarray, template_mask: np.ndarray, base: int) -> list[dict]:
    void_mask = ~template_mask
    cavities: list[dict] = []
    for coords in components_mask(void_mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        local = np.zeros((h, w), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        assert local.all()
        residue = crop[r0 : r1 + 1, c0 : c1 + 1] == int(base)
        assert residue.any() and (~residue).any()
        cavities.append(
            {
                "bbox": (int(r0), int(c0), h, w),
                "residue_mask": residue,
            }
        )
    return cavities


def _extract_sprites(grid: np.ndarray, bg: int, template_bbox: tuple[int, int, int, int]) -> list[dict]:
    masked = grid.copy()
    r0, c0, r1, c1 = template_bbox
    masked[r0 : r1 + 1, c0 : c1 + 1] = bg
    comps = find_components(masked, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    sprites: list[dict] = []
    for comp in comps:
        br0, bc0, br1, bc1 = map(int, comp["bbox"])
        patch = crop_bbox(masked, (br0, bc0, br1, bc1))
        cols = colors_present(patch, exclude={bg})
        assert len(cols) == 1
        color = int(cols[0])
        mask = patch == color
        assert _mask_connected(mask)
        sprites.append(
            {
                "bbox": (br0, bc0, br1, bc1),
                "shape": mask,
                "color": color,
            }
        )
    return sprites


def main(input_grid: np.ndarray) -> np.ndarray:
    """Place each outside sprite into the same-sized cavity whose residue is its exact negative.

    Input structure:
    - A large base-color template sits on the background, and its main connected frame contains several internal rectangular cavities.
    - Each cavity is partly empty background and partly residual base-color pixels.
    - Outside the template, one monochrome source sprite exists for each cavity, and each sprite's bbox matches its target cavity's rectangle size.

    Rule:
    - Identify the template as the unique largest non-background monochrome component and crop to its bbox.
    - Inside that crop, every connected region not belonging to the main template frame is one rectangular cavity; keep its residual base-color pattern.
    - Match each outside sprite to the unique cavity of the same size whose residual pattern is the exact complement of the sprite mask, then paste the sprite into the cavity's background cells.

    Key insight:
    - Equal rectangle size is intentionally ambiguous; the decisive key is pixelwise complementarity between a cavity's residue and an external sprite's negative mask.
    """
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2

    bg = _border_background(g)
    base, coords = _largest_template_component(g, bg)
    tr0, tc0, tr1, tc1 = bbox_coords(coords)
    crop = crop_bbox(g, (tr0, tc0, tr1, tc1))

    template_mask = np.zeros(crop.shape, dtype=bool)
    template_mask[coords[:, 0] - tr0, coords[:, 1] - tc0] = True
    cavities = _extract_cavities(crop, template_mask, base)
    sprites = _extract_sprites(g, bg, (tr0, tc0, tr1, tc1))
    assert len(cavities) == len(sprites) and cavities

    out = crop.copy()
    used_cavities: set[int] = set()
    for sprite in sprites:
        h, w = sprite["shape"].shape
        matches = [
            idx
            for idx, cavity in enumerate(cavities)
            if idx not in used_cavities
            and cavity["bbox"][2:] == (h, w)
            and np.array_equal(sprite["shape"], ~cavity["residue_mask"])
        ]
        assert len(matches) == 1
        idx = matches[0]
        used_cavities.add(idx)
        r0, c0, _h, _w = cavities[idx]["bbox"]
        patch = out[r0 : r0 + h, c0 : c0 + w]
        assert np.all(patch[sprite["shape"]] == bg)
        patch[sprite["shape"]] = int(sprite["color"])

    assert len(used_cavities) == len(cavities)
    return out


def _generate_instance() -> tuple[np.ndarray, np.ndarray]:
    built = _build_template_instance()
    if built is None:
        raise RuntimeError("template generation failed")
    template, cavities, bg, base = built
    expected = _expected_output(template, cavities, base)

    H, W, top, left = _choose_canvas(*template.shape)
    grid = np.full((H, W), bg, dtype=int)
    paste(grid, template, top, left)
    ok = _place_sources(grid, (top, left, template.shape[0], template.shape[1]), cavities, bg)
    if not ok:
        raise RuntimeError("source placement failed")
    return grid, expected


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> None:
    out = main(inp)
    assert isinstance(out, np.ndarray) and out.ndim == 2
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)

    bg = _border_background(inp)
    base, coords = _largest_template_component(inp, bg)
    tr0, tc0, tr1, tc1 = bbox_coords(coords)
    crop = crop_bbox(inp, (tr0, tc0, tr1, tc1))
    template_mask = np.zeros(crop.shape, dtype=bool)
    template_mask[coords[:, 0] - tr0, coords[:, 1] - tc0] = True
    cavities = _extract_cavities(crop, template_mask, base)
    sprites = _extract_sprites(inp, bg, (tr0, tc0, tr1, tc1))

    assert len(cavities) == len(sprites) >= 3
    repeated = False
    by_size: dict[tuple[int, int], list[np.ndarray]] = {}
    for cavity in cavities:
        size = cavity["bbox"][2:]
        by_size.setdefault(size, []).append(cavity["residue_mask"])
        if len(by_size[size]) >= 2:
            repeated = True
    assert repeated

    for sprite in sprites:
        matches = [
            idx
            for idx, cavity in enumerate(cavities)
            if cavity["bbox"][2:] == sprite["shape"].shape
            and np.array_equal(sprite["shape"], ~cavity["residue_mask"])
        ]
        assert len(matches) == 1

    size_only_ambiguous = any(len(group) >= 2 for group in by_size.values())
    assert size_only_ambiguous


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, expected = _generate_instance()
            _semantic_checks(inp, expected)
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid negative-inlay puzzle")


if __name__ == "__main__":
    for _ in range(10):
        grid = generate_input()
        result = main(grid)
        assert grid.ndim == 2 and result.ndim == 2
        assert max(grid.shape) <= 30 and max(result.shape) <= 30
