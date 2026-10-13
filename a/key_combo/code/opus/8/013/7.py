from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("arc_utils_shared_013_7", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load arc_utils from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color
bbox = _ARC.bbox
bbox_h = _ARC.bbox_h
bbox_w = _ARC.bbox_w
colors_present = _ARC.colors_present
crop_bbox = _ARC.crop_bbox
find_components = _ARC.find_components
get_bg = _ARC.get_bg
paste = _ARC.paste


GENERATION_ATTEMPTS = 400
SPRITE_COUNT_CHOICES = (3, 4)
SPRITE_COLOR_POOL = (2, 3, 6, 7, 9)
MARKER_COLOR_POOL = (1, 4, 5, 8)
DISTANCE_POOL = (1, 2, 3, 4, 5)


def _component_mask(component: dict, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _extract_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    markers: list[dict] = []
    sprites: list[dict] = []
    for component in components:
        color = int(component["color"])
        area = int(component["area"])
        b = tuple(int(x) for x in component["bbox"])
        coords = np.asarray(component["coords"], dtype=int)

        if area == 1 and int(coords[0, 1]) == 0:
            markers.append({"color": color, "row": int(coords[0, 0]), "col": 0})
            continue

        assert b[1] > 0, "sprites must stay to the right of the marker column"
        sprite_crop = np.where(crop_bbox(g, b) == color, color, bg)
        sprite_mask = sprite_crop == color
        assert sprite_mask.any()
        sprites.append(
            {
                "color": color,
                "bbox": b,
                "sprite": sprite_crop,
                "mask": sprite_mask,
                "height": int(bbox_h(b)),
                "width": int(bbox_w(b)),
            }
        )

    assert markers, "expected left-edge singleton markers"
    assert sprites, "expected stacked sprites"

    markers.sort(key=lambda item: item["row"])
    sprites.sort(key=lambda item: item["bbox"][0])

    assert len({m["color"] for m in markers}) == len(markers), "marker colors must be distinct"
    assert len({s["color"] for s in sprites}) == len(sprites), "sprite colors must be distinct"
    assert not ({m["color"] for m in markers} & {s["color"] for s in sprites}), "marker and sprite colors must differ"
    assert len(markers) == len(sprites), "each sprite needs exactly one marker"

    for prev, nxt in zip(sprites, sprites[1:]):
        assert int(prev["bbox"][2]) + 1 == int(nxt["bbox"][0]), "sprite bboxes must form a contiguous vertical stack"

    return bg, markers, sprites


def _pair_markers_against_grid(
    grid: np.ndarray, bg: int, markers: list[dict], sprite_colors: set[int]
) -> dict[int, dict]:
    pairings: dict[int, dict] = {}
    distances: list[int] = []
    H, W = grid.shape

    for marker in markers:
        r = int(marker["row"])
        assert 0 <= r < H
        hit = None
        for c in range(1, W):
            if int(grid[r, c]) != bg:
                hit = (c, int(grid[r, c]))
                break
        assert hit is not None, "each marker row must see a sprite to the right"
        hit_c, hit_color = hit
        assert hit_color in sprite_colors, "marker ray must hit a sprite"
        assert hit_color not in pairings, "markers must pair one-to-one with sprites"
        pairings[hit_color] = {
            "marker_color": int(marker["color"]),
            "distance": int(hit_c - 1),
            "marker_row": r,
        }
        distances.append(int(hit_c - 1))

    assert len(pairings) == len(sprite_colors)
    assert len(set(distances)) == len(distances), "marker distances must be distinct"
    return pairings


def _render_stack(
    shape: tuple[int, int],
    bg: int,
    markers: list[dict],
    sprites: list[dict],
    *,
    reverse: bool,
) -> tuple[np.ndarray, dict[int, dict]]:
    out = np.full(shape, bg, dtype=int)
    for marker in markers:
        out[int(marker["row"]), 0] = int(marker["color"])

    placements: dict[int, dict] = {}
    cursor = int(sprites[0]["bbox"][0])
    ordered = list(reversed(sprites)) if reverse else list(sprites)
    for sprite in ordered:
        r0, c0, _r1, c1 = sprite["bbox"]
        paste(out, sprite["sprite"], cursor, int(c0), transparent=bg)
        placements[int(sprite["color"])] = {
            "r0": int(cursor),
            "c0": int(c0),
            "c1": int(c1),
            "mask": sprite["mask"],
            "height": int(sprite["height"]),
        }
        cursor += int(sprite["height"])

    assert cursor == int(sprites[-1]["bbox"][2]) + 1
    return out, placements


def _apply_recolor(
    grid: np.ndarray, placements: dict[int, dict], pairings: dict[int, dict]
) -> np.ndarray:
    out = grid.copy()
    for color, placement in placements.items():
        marker_color = int(pairings[color]["marker_color"])
        distance = int(pairings[color]["distance"])
        mask = np.asarray(placement["mask"], dtype=bool)
        height = int(placement["height"])
        start_local_r = max(0, height - distance)
        for local_r in range(start_local_r, height):
            row_view = out[int(placement["r0"]) + local_r, int(placement["c0"]) : int(placement["c1"]) + 1]
            row_view[mask[local_r]] = marker_color
    return out


def _solve_with_pairings(
    grid: np.ndarray,
    bg: int,
    markers: list[dict],
    sprites: list[dict],
    pairings: dict[int, dict],
    *,
    reverse: bool,
    recolor: bool,
) -> np.ndarray:
    rendered, placements = _render_stack(grid.shape, bg, markers, sprites, reverse=reverse)
    if not recolor:
        return rendered
    return _apply_recolor(rendered, placements, pairings)


def _pair_after_reverse(grid: np.ndarray, bg: int, markers: list[dict], sprites: list[dict]) -> np.ndarray:
    rendered, _placements = _render_stack(grid.shape, bg, markers, sprites, reverse=True)
    pairings = _pair_markers_against_grid(rendered, bg, markers, {int(s["color"]) for s in sprites})
    return _apply_recolor(rendered, _placements, pairings)


def _solve_scene(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, markers, sprites = _extract_scene(g)
    pairings = _pair_markers_against_grid(g, bg, markers, {int(s["color"]) for s in sprites})
    return _solve_with_pairings(g, bg, markers, sprites, pairings, reverse=True, recolor=True)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair left-edge markers to visible stacked sprites, reverse the stack, then repaint each new tail by the pre-reversal sightline distance.

    Input structure:
    - Black background with one singleton marker on column 0 for each sprite.
    - Distinct-color sprites on the right form a contiguous vertical stack of 4-connected irregular bodies.
    - Every marker lies on a row that horizontally sees exactly one sprite, and the background gap before that first hit is meaningful.

    Rule:
    - For each marker, cast a horizontal ray rightward along its row and pair it with the first sprite pixel reached.
    - Record the number of background cells crossed before that hit.
    - Reverse the top-to-bottom order of the sprite stack while preserving each sprite's internal shape and columns.
    - In the reversed position, repaint the bottom D rows of each paired sprite with the marker color, where D is that marker's recorded gap; keep the markers themselves unchanged.

    Key insight:
    - Pairing and distance must be captured before the stack is reversed, because the marker decides both which sprite is affected and how many tail rows are recolored after relocation.
    """
    return _solve_scene(np.asarray(grid, dtype=int))


def _connected(mask: np.ndarray) -> bool:
    comps = find_components(mask.astype(int), bg=0, connectivity=4, monochromatic=True, return_format="dict")
    return len(comps) == 1


def _make_sprite_mask(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    removable = [(r, c) for r in range(height) for c in range(1, width)]
    max_remove = max(1, min(len(removable) - 1, (height * width) // 3))

    for _ in range(120):
        mask = np.ones((height, width), dtype=bool)
        target_remove = int(rng.integers(1, max_remove + 1))
        protected_last_col = (int(rng.integers(0, height)), width - 1)
        order = removable.copy()
        rng.shuffle(order)

        removed = 0
        for r, c in order:
            if (r, c) == protected_last_col:
                continue
            trial = mask.copy()
            trial[r, c] = False
            if not trial[:, 0].all():
                continue
            if not trial[:, -1].any():
                continue
            if np.any(trial.sum(axis=1) == 0):
                continue
            if not _connected(trial):
                continue
            mask = trial
            removed += 1
            if removed >= target_remove:
                break

        if removed >= 1 and not mask.all():
            return mask

    raise RuntimeError("failed to construct irregular sprite mask")


def _sample_distances(rng: np.random.Generator, count: int) -> list[int]:
    while True:
        distances = sorted(int(x) for x in rng.choice(DISTANCE_POOL, size=count, replace=False))
        if min(distances) <= 2 and max(distances) >= 4:
            return distances


def _build_scene(rng: np.random.Generator) -> np.ndarray:
    count = int(rng.choice(SPRITE_COUNT_CHOICES))
    sprite_colors = list(int(x) for x in rng.choice(SPRITE_COLOR_POOL, size=count, replace=False))
    marker_colors = list(int(x) for x in rng.choice(MARKER_COLOR_POOL, size=count, replace=False))
    distances = _sample_distances(rng, count)
    rng.shuffle(sprite_colors)
    rng.shuffle(marker_colors)
    rng.shuffle(distances)

    while True:
        heights = [int(rng.integers(2, 5)) for _ in range(count)]
        if any(d < h for d, h in zip(distances, heights)) and any(d >= h for d, h in zip(distances, heights)):
            break
    widths = [int(rng.integers(3, 7)) for _ in range(count)]

    stack_height = sum(heights)
    top_pad = int(rng.integers(1, 4))
    bottom_pad = int(rng.integers(1, 4))
    H = top_pad + stack_height + bottom_pad
    rightmost = max((d + 1) + w - 1 for d, w in zip(distances, widths))
    W = max(14, rightmost + 1 + int(rng.integers(1, 4)))
    assert H <= 30 and W <= 30

    grid = np.zeros((H, W), dtype=int)
    row = top_pad
    for sprite_color, marker_color, distance, height, width in zip(
        sprite_colors, marker_colors, distances, heights, widths
    ):
        mask = _make_sprite_mask(rng, height, width)
        c0 = distance + 1
        patch = np.where(mask, sprite_color, 0)
        paste(grid, patch, row, c0, transparent=0)
        marker_row = row + int(rng.integers(0, height))
        grid[marker_row, 0] = marker_color
        row += height

    return grid


def _validate_tail_recolor(grid: np.ndarray, output: np.ndarray) -> bool:
    bg, markers, sprites = _extract_scene(grid)
    pairings = _pair_markers_against_grid(grid, bg, markers, {int(s["color"]) for s in sprites})
    rendered, placements = _render_stack(grid.shape, bg, markers, sprites, reverse=True)

    for color, placement in placements.items():
        marker_color = int(pairings[color]["marker_color"])
        distance = int(pairings[color]["distance"])
        height = int(placement["height"])
        c0 = int(placement["c0"])
        c1 = int(placement["c1"])
        start = max(0, height - distance)
        mask = np.asarray(placement["mask"], dtype=bool)
        for local_r in range(height):
            want_color = marker_color if local_r >= start else color
            row_out = output[int(placement["r0"]) + local_r, c0 : c1 + 1]
            row_base = rendered[int(placement["r0"]) + local_r, c0 : c1 + 1]
            assert np.array_equal(row_out[~mask[local_r]], row_base[~mask[local_r]])
            if np.any(mask[local_r]):
                assert np.all(row_out[mask[local_r]] == want_color)

    for marker in markers:
        assert int(output[int(marker["row"]), 0]) == int(marker["color"])
    return True


def _is_valid_example(grid: np.ndarray) -> bool:
    try:
        bg, markers, sprites = _extract_scene(grid)
        pairings = _pair_markers_against_grid(grid, bg, markers, {int(s["color"]) for s in sprites})
        if len(sprites) not in SPRITE_COUNT_CHOICES:
            return False

        heights = [int(sprite["height"]) for sprite in sprites]
        distances = [int(pairings[int(sprite["color"])]["distance"]) for sprite in sprites]
        if not any(d < h for d, h in zip(distances, heights)):
            return False
        if not any(d >= h for d, h in zip(distances, heights)):
            return False

        correct = _solve_with_pairings(grid, bg, markers, sprites, pairings, reverse=True, recolor=True)
        no_reverse = _solve_with_pairings(grid, bg, markers, sprites, pairings, reverse=False, recolor=True)
        reverse_plain = _solve_with_pairings(grid, bg, markers, sprites, pairings, reverse=True, recolor=False)
        pair_after_reverse = _pair_after_reverse(grid, bg, markers, sprites)

        if np.array_equal(correct, grid):
            return False
        if np.array_equal(correct, no_reverse):
            return False
        if np.array_equal(correct, reverse_plain):
            return False
        if np.array_equal(correct, pair_after_reverse):
            return False

        _validate_tail_recolor(grid, correct)
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_scene(rng)
        if _is_valid_example(grid):
            return grid
    raise RuntimeError("failed to generate a valid stacked-sightline ARC input")
