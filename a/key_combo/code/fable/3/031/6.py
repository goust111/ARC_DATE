from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste, random_free_location, random_sprite


BG = Color.BLACK
TRANSFORM_KINDS = ["rot90", "rot180", "rot270", "flip_lr", "flip_ud", "swap"]
GEN_RETRIES = 400
_TRANSFORM_BAG: list[str] = []


def _apply_transform(sprite: np.ndarray, spec: tuple) -> np.ndarray:
    kind = spec[0]
    g = np.asarray(sprite, dtype=int)
    if kind == "rot90":
        return np.rot90(g, 1).copy()
    if kind == "rot180":
        return np.rot90(g, 2).copy()
    if kind == "rot270":
        return np.rot90(g, 3).copy()
    if kind == "flip_lr":
        return np.fliplr(g).copy()
    if kind == "flip_ud":
        return np.flipud(g).copy()
    if kind == "swap":
        a, b = map(int, spec[1:3])
        out = g.copy()
        mask_a = g == a
        mask_b = g == b
        out[mask_a] = b
        out[mask_b] = a
        return out
    raise ValueError(f"unknown transform kind: {kind}")


def _matching_transforms(before: np.ndarray, after: np.ndarray, bg: int) -> list[tuple]:
    matches: list[tuple] = []
    for kind in TRANSFORM_KINDS:
        if kind == "swap":
            colors = sorted(set(map(int, np.unique(before))) - {int(bg)})
            for i, a in enumerate(colors):
                for b in colors[i + 1 :]:
                    spec = ("swap", int(a), int(b))
                    if np.array_equal(_apply_transform(before, spec), after):
                        matches.append(spec)
            continue
        spec = (kind,)
        transformed = _apply_transform(before, spec)
        if transformed.shape == after.shape and np.array_equal(transformed, after):
            matches.append(spec)
    return matches


def _infer_transform(before: np.ndarray, after: np.ndarray, bg: int) -> tuple:
    matches = _matching_transforms(before, after, bg)
    assert len(matches) == 1, f"expected one demo transform, found {matches}"
    return matches[0]


def _region_components(grid: np.ndarray, bounds: tuple[int, int, int, int], bg: int) -> list[dict]:
    r0, r1, c0, c1 = map(int, bounds)
    if r0 >= r1 or c0 >= c1:
        return []
    sub = grid[r0:r1, c0:c1]
    comps = find_components(
        sub,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    out: list[dict] = []
    for comp in comps:
        br0, bc0, br1, bc1 = map(int, comp["bbox"])
        out.append(
            {
                "bbox": (r0 + br0, c0 + bc0, r0 + br1, c0 + bc1),
                "sprite": crop_bbox(sub, (br0, bc0, br1, bc1)),
            }
        )
    return out


def _evaluate_partition(
    grid: np.ndarray,
    bg: int,
    demo_bounds: tuple[int, int, int, int],
    main_bounds: tuple[int, int, int, int],
) -> dict | None:
    demo_comps = _region_components(grid, demo_bounds, bg)
    main_comps = _region_components(grid, main_bounds, bg)
    if len(demo_comps) != 2 or len(main_comps) < 2:
        return None

    demo_comps = sorted(demo_comps, key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    left_box = demo_comps[0]["bbox"]
    right_box = demo_comps[1]["bbox"]
    if left_box[3] >= right_box[1]:
        return None

    shape = demo_comps[0]["sprite"].shape
    if demo_comps[1]["sprite"].shape != shape:
        return None
    if any(comp["sprite"].shape != shape for comp in main_comps):
        return None

    try:
        spec = _infer_transform(demo_comps[0]["sprite"], demo_comps[1]["sprite"], bg)
    except AssertionError:
        return None

    demo_area = (demo_bounds[1] - demo_bounds[0]) * (demo_bounds[3] - demo_bounds[2])
    main_area = (main_bounds[1] - main_bounds[0]) * (main_bounds[3] - main_bounds[2])
    return {
        "spec": spec,
        "main_comps": sorted(main_comps, key=lambda comp: (comp["bbox"][0], comp["bbox"][1])),
        "score": (demo_area < main_area, len(main_comps), -demo_area),
    }


def _find_layout(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape
    candidates: list[dict] = []

    for r in range(h):
        line_color = int(g[r, 0])
        if line_color == bg or not np.all(g[r, :] == line_color):
            continue
        top = (0, r, 0, w)
        bottom = (r + 1, h, 0, w)
        for demo_bounds, main_bounds in ((top, bottom), (bottom, top)):
            candidate = _evaluate_partition(g, bg, demo_bounds, main_bounds)
            if candidate is not None:
                candidates.append(candidate)

    for c in range(w):
        line_color = int(g[0, c])
        if line_color == bg or not np.all(g[:, c] == line_color):
            continue
        left = (0, h, 0, c)
        right = (0, h, c + 1, w)
        for demo_bounds, main_bounds in ((left, right), (right, left)):
            candidate = _evaluate_partition(g, bg, demo_bounds, main_bounds)
            if candidate is not None:
                candidates.append(candidate)

    assert candidates, "failed to locate separator and demo pair"
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the embedded before/after transform and apply it in place to every same-sized sprite in the main region.

    Input structure:
    - A solid separator row or column splits the grid into a small demo region and a larger main region.
    - The demo region contains two same-sized multicolor sprites placed side by side; the left sprite is the pre-transform example and the right sprite is the post-transform example.
    - The main region contains several other sprites with the same tight bbox size, separated from one another by background.

    Rule:
    - Compare the demo pair and find the unique transformation that maps the left sprite to the right one; the transformation may be a 90/180/270 degree rotation, a horizontal or vertical mirror, or a swap of one foreground color pair.
    - For each sprite in the main region, crop its own tight bbox, apply that same transformation inside the bbox, and write the transformed sprite back at the original location.
    - Leave the demo pair, the separator, and all background elsewhere unchanged.

    Key insight:
    - No fixed transform is shared across instances; the instruction is encoded by the input's own before/after demo pair, so the solver must decode that local example and transfer it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    layout = _find_layout(g)
    spec = layout["spec"]
    bg = int(get_bg(g))
    out = g.copy()

    for comp in layout["main_comps"]:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        sprite = crop_bbox(g, (r0, c0, r1, c1))
        transformed = _apply_transform(sprite, spec)
        out[r0 : r1 + 1, c0 : c1 + 1] = bg
        paste(out, transformed, r0, c0, transparent=bg)

    return out


def _next_transform_kind() -> str:
    global _TRANSFORM_BAG
    if not _TRANSFORM_BAG:
        _TRANSFORM_BAG = random.sample(TRANSFORM_KINDS, k=len(TRANSFORM_KINDS))
    return _TRANSFORM_BAG.pop()


def _random_pattern(
    h: int,
    w: int,
    palette: list[int],
    *,
    required_colors: set[int] | None = None,
    changed_by: tuple | None = None,
) -> np.ndarray:
    required = set() if required_colors is None else {int(c) for c in required_colors}
    min_colors = min(2, len(set(palette)))
    for _ in range(200):
        sprite = random_sprite(
            h,
            w,
            density=random.uniform(0.35, 0.7),
            symmetry="not_symmetric",
            color_palette=list(palette),
            connectivity=4,
            background=BG,
        ).astype(int)
        present = set(map(int, np.unique(sprite))) - {BG}
        if len(present) < min_colors or not required.issubset(present):
            continue
        if changed_by is not None and np.array_equal(_apply_transform(sprite, changed_by), sprite):
            continue
        return sprite
    raise RuntimeError("failed to generate a non-degenerate sprite")


def _pick_demo_spec(kind: str) -> tuple[np.ndarray, np.ndarray, tuple, list[int]]:
    palette_pool = [c for c in Color.NOT_BLACK]
    for _ in range(200):
        if kind in {"rot90", "rot270"}:
            size = random.randint(3, 6)
            h = w = size
        else:
            h = random.randint(3, 6)
            w = random.randint(3, 6)

        if kind == "swap":
            palette_size = random.choice([2, 3])
            palette = random.sample(palette_pool, k=palette_size)
            a, b = sorted(random.sample(palette, k=2))
            spec = ("swap", int(a), int(b))
            sprite = _random_pattern(h, w, palette, required_colors={a, b}, changed_by=spec)
        else:
            palette_size = random.randint(2, 4)
            palette = random.sample(palette_pool, k=palette_size)
            spec = (kind,)
            sprite = _random_pattern(h, w, palette, changed_by=spec)

        after = _apply_transform(sprite, spec)
        if _matching_transforms(sprite, after, BG) == [spec]:
            return sprite, after, spec, palette
    raise RuntimeError("failed to build a unique demo transform")


def _make_main_sprites(shape: tuple[int, int], palette: list[int], spec: tuple, count: int, demo: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, shape)
    required: set[int] = set()
    if spec[0] == "swap":
        required = {int(spec[1]), int(spec[2])}

    seen = {demo.tobytes()}
    sprites: list[np.ndarray] = []
    for _ in range(count):
        for _ in range(200):
            sprite = _random_pattern(h, w, palette, required_colors=required, changed_by=spec)
            key = sprite.tobytes()
            if key in seen:
                continue
            seen.add(key)
            sprites.append(sprite)
            break
        else:
            raise RuntimeError("failed to build distinct main sprites")
    return sprites


def _place_demo_pair(
    grid: np.ndarray,
    demo_before: np.ndarray,
    demo_after: np.ndarray,
    bounds: tuple[int, int, int, int],
    side: str,
) -> None:
    r0, r1, c0, c1 = map(int, bounds)
    sub_h = r1 - r0
    sub_w = c1 - c0
    h, w = demo_before.shape
    gap = random.randint(1, 3)
    pair_w = 2 * w + gap

    if side == "top":
        rr = 0
        cc = 0 if random.random() < 0.5 else sub_w - pair_w
    elif side == "bottom":
        rr = sub_h - h
        cc = 0 if random.random() < 0.5 else sub_w - pair_w
    elif side == "left":
        rr = 0 if random.random() < 0.5 else sub_h - h
        cc = 0
    elif side == "right":
        rr = 0 if random.random() < 0.5 else sub_h - h
        cc = sub_w - pair_w
    else:
        raise ValueError(f"unknown demo side: {side}")

    paste(grid, demo_before, r0 + rr, c0 + cc, transparent=BG)
    paste(grid, demo_after, r0 + rr, c0 + cc + w + gap, transparent=BG)


def _place_main_sprites(
    grid: np.ndarray,
    sprites: list[np.ndarray],
    bounds: tuple[int, int, int, int],
) -> list[tuple[int, int, np.ndarray]]:
    r0, r1, c0, c1 = map(int, bounds)
    sub = np.full((r1 - r0, c1 - c0), BG, dtype=int)
    placements: list[tuple[int, int, np.ndarray]] = []
    for sprite in random.sample(sprites, k=len(sprites)):
        loc_r, loc_c = random_free_location(
            sub,
            sprite,
            transparent=BG,
            border=1,
            padding=random.choice([1, 2]),
        )
        paste(sub, sprite, loc_r, loc_c, transparent=BG)
        placements.append((r0 + loc_r, c0 + loc_c, sprite))
    grid[r0:r1, c0:c1] = sub
    return placements


def _build_input() -> tuple[np.ndarray, np.ndarray]:
    kind = _next_transform_kind()
    demo_before, demo_after, spec, palette = _pick_demo_spec(kind)
    n_main = random.randint(2, 4)
    main_sprites = _make_main_sprites(demo_before.shape, palette, spec, n_main, demo_before)

    gap = random.randint(1, 3)
    pair_w = 2 * demo_before.shape[1] + gap
    side = random.choice(["top", "bottom", "left", "right"])
    sep_color = random.choice([c for c in Color.NOT_BLACK if c not in palette])

    for _ in range(200):
        if side in {"top", "bottom"}:
            demo_band_h = demo_before.shape[0] + random.randint(2, 4)
            min_w = max(pair_w, 14)
            height = random.randint(demo_band_h + 1 + max(demo_before.shape[0] + 6, 10), 30)
            width = random.randint(min_w, 30)
            grid = np.full((height, width), BG, dtype=int)
            if side == "top":
                sep = demo_band_h
                demo_bounds = (0, sep, 0, width)
                main_bounds = (sep + 1, height, 0, width)
            else:
                sep = height - demo_band_h - 1
                main_bounds = (0, sep, 0, width)
                demo_bounds = (sep + 1, height, 0, width)
            grid[sep, :] = sep_color
        else:
            demo_band_w = pair_w + random.randint(0, 2)
            min_h = max(demo_before.shape[0], 14)
            height = random.randint(min_h, 30)
            width = random.randint(demo_band_w + 1 + max(demo_before.shape[1] + 6, 10), 30)
            grid = np.full((height, width), BG, dtype=int)
            if side == "left":
                sep = demo_band_w
                demo_bounds = (0, height, 0, sep)
                main_bounds = (0, height, sep + 1, width)
            else:
                sep = width - demo_band_w - 1
                main_bounds = (0, height, 0, sep)
                demo_bounds = (0, height, sep + 1, width)
            grid[:, sep] = sep_color

        try:
            _place_demo_pair(grid, demo_before, demo_after, demo_bounds, side)
            placements = _place_main_sprites(grid, main_sprites, main_bounds)
        except ValueError:
            continue

        expected = grid.copy()
        for r0, c0, sprite in placements:
            transformed = _apply_transform(sprite, spec)
            h, w = sprite.shape
            expected[r0 : r0 + h, c0 : c0 + w] = BG
            paste(expected, transformed, r0, c0, transparent=BG)

        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid, expected

    raise RuntimeError("failed to assemble a valid puzzle instance")


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, _expected = _build_input()
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
