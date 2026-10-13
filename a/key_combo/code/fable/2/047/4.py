from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(_TOOLBOX_DIR))

from arc_utils import Color, find_components, get_bg, paste, random_free_location


BG = Color.BLACK
TRANSFORMS = (
    "id",
    "rot90",
    "rot180",
    "rot270",
    "flip",
    "flip_rot90",
    "flip_rot180",
    "flip_rot270",
)
NON_ID_TRANSFORMS = TRANSFORMS[1:]

GEN_MAX_ATTEMPTS = 500
GRID_ROWS = (18, 26)
GRID_COLS = (18, 26)
SHAPE_H = (3, 6)
SHAPE_W = (3, 6)
SHAPE_AREA = (5, 12)
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
RECENT_OUTPUT_LIMIT = 16

_TRANSFORM_CYCLE: list[str] = []
_DISTRACTOR_CYCLE: list[str] = []
_RECENT_OUTPUT_SIGS: list[tuple[tuple[int, int], bytes]] = []


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(mask, dtype=np.uint8)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _apply_transform(mask: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if name == "id":
        return arr.copy()
    if name == "rot90":
        return np.rot90(arr, 1).copy()
    if name == "rot180":
        return np.rot90(arr, 2).copy()
    if name == "rot270":
        return np.rot90(arr, 3).copy()
    if name == "flip":
        return np.fliplr(arr).copy()
    if name == "flip_rot90":
        return np.fliplr(np.rot90(arr, 1)).copy()
    if name == "flip_rot180":
        return np.fliplr(np.rot90(arr, 2)).copy()
    if name == "flip_rot270":
        return np.fliplr(np.rot90(arr, 3)).copy()
    raise ValueError(f"unknown transform: {name}")


def _orbit_keys(mask: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    return {_mask_key(_apply_transform(mask, name)) for name in TRANSFORMS}


def _find_unique_transform(src: np.ndarray, dst: np.ndarray) -> str | None:
    matches = [name for name in TRANSFORMS if np.array_equal(_apply_transform(src, name), dst)]
    if len(matches) != 1:
        return None
    return matches[0]


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, component["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = np.asarray(component["coords"], dtype=int)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    assert int(bg) == int(BG), "expected black background"

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert len(comps) == 3, "expected exactly three sprites"

    items = []
    for comp in comps:
        items.append(
            {
                "color": int(comp["color"]),
                "bbox": tuple(map(int, comp["bbox"])),
                "mask": _component_mask(comp),
            }
        )

    color_counts = Counter(item["color"] for item in items)
    assert sorted(color_counts.values()) == [1, 2], "one visible color must repeat to mark the source"

    iso_pairs = []
    for i in range(3):
        for j in range(i + 1, 3):
            transform = _find_unique_transform(items[i]["mask"], items[j]["mask"])
            if transform is not None:
                iso_pairs.append((i, j, transform))
    assert len(iso_pairs) == 1, "expected exactly one isomorphic pair"

    i, j, _unused = iso_pairs[0]
    target_idx = next(idx for idx in range(3) if idx not in (i, j))
    target_color = items[target_idx]["color"]
    source_candidates = [idx for idx in (i, j) if items[idx]["color"] == target_color]
    assert len(source_candidates) == 1, "the repeated color must identify the demo source"

    source_idx = source_candidates[0]
    image_idx = j if source_idx == i else i
    transform = _find_unique_transform(items[source_idx]["mask"], items[image_idx]["mask"])
    assert transform is not None and transform != "id", "the demo transform must be uniquely non-trivial"

    out_mask = _apply_transform(items[target_idx]["mask"], transform)
    out = np.full(out_mask.shape, int(bg), dtype=int)
    out[out_mask] = items[image_idx]["color"]

    return {
        "components": items,
        "source_idx": int(source_idx),
        "image_idx": int(image_idx),
        "target_idx": int(target_idx),
        "source_color": int(items[source_idx]["color"]),
        "image_color": int(items[image_idx]["color"]),
        "transform": transform,
        "output": out,
    }


def _grow_connected_mask(height: int, width: int, area: int) -> np.ndarray | None:
    h = int(height)
    w = int(width)
    area = int(area)
    if h <= 0 or w <= 0 or area <= 0 or area > h * w:
        return None

    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = {
        nbr
        for nbr in ((start[0] - 1, start[1]), (start[0] + 1, start[1]), (start[0], start[1] - 1), (start[0], start[1] + 1))
        if 0 <= nbr[0] < h and 0 <= nbr[1] < w
    }
    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nbr in ((cell[0] - 1, cell[1]), (cell[0] + 1, cell[1]), (cell[0], cell[1] - 1), (cell[0], cell[1] + 1)):
            if 0 <= nbr[0] < h and 0 <= nbr[1] < w and nbr not in cells:
                frontier.add(nbr)
    if len(cells) != area:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_connected_mask(
    *,
    exact_area: int | None = None,
    exact_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    for _ in range(120):
        if exact_shape is None:
            h = random.randint(*SHAPE_H)
            w = random.randint(*SHAPE_W)
        else:
            h, w = map(int, exact_shape)

        max_area = min(int(h * w) - 1, SHAPE_AREA[1])
        min_area = SHAPE_AREA[0]
        if exact_area is not None:
            area = int(exact_area)
            if area < min_area or area > max_area:
                continue
        else:
            if max_area < min_area:
                continue
            area = random.randint(min_area, max_area)

        mask = _grow_connected_mask(h, w, area)
        if mask is None:
            continue
        if exact_shape is not None and tuple(map(int, mask.shape)) != tuple(map(int, exact_shape)):
            continue
        return mask
    return None


def _make_asymmetric_mask(
    *,
    forbidden_orbit: set[tuple[tuple[int, int], bytes]] | None = None,
    exact_area: int | None = None,
    exact_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    for _ in range(240):
        mask = _random_connected_mask(exact_area=exact_area, exact_shape=exact_shape)
        if mask is None:
            continue
        if min(mask.shape) < 3:
            continue
        if int(mask.sum()) >= int(mask.size):
            continue
        orbit = _orbit_keys(mask)
        if len(orbit) != len(TRANSFORMS):
            continue
        if forbidden_orbit is not None and orbit & forbidden_orbit:
            continue
        return mask
    return None


def _render_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(color)
    return patch


def _next_transform() -> str:
    global _TRANSFORM_CYCLE
    if not _TRANSFORM_CYCLE:
        _TRANSFORM_CYCLE = list(NON_ID_TRANSFORMS)
        random.shuffle(_TRANSFORM_CYCLE)
    return _TRANSFORM_CYCLE.pop()


def _next_distractor_mode() -> str:
    global _DISTRACTOR_CYCLE
    if not _DISTRACTOR_CYCLE:
        _DISTRACTOR_CYCLE = ["same_area", "same_bbox", "free"]
        random.shuffle(_DISTRACTOR_CYCLE)
    return _DISTRACTOR_CYCLE.pop()


def _output_signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(grid, dtype=np.uint8)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _remember_output(sig: tuple[tuple[int, int], bytes]) -> None:
    _RECENT_OUTPUT_SIGS.append(sig)
    if len(_RECENT_OUTPUT_SIGS) > RECENT_OUTPUT_LIMIT:
        del _RECENT_OUTPUT_SIGS[0]


def _validate_generated(
    grid: np.ndarray,
    *,
    transform: str,
    source_color: int,
    image_color: int,
    source_orbit: set[tuple[tuple[int, int], bytes]],
    target_orbit: set[tuple[tuple[int, int], bytes]],
    expected: np.ndarray,
) -> bool:
    if source_orbit & target_orbit:
        return False
    try:
        meta = _analyze_scene(grid)
    except AssertionError:
        return False

    source_mask = meta["components"][meta["source_idx"]]["mask"]
    target_mask = meta["components"][meta["target_idx"]]["mask"]
    recolor_only = _render_patch(target_mask, image_color)

    if meta["transform"] != transform:
        return False
    if meta["source_color"] != int(source_color) or meta["image_color"] != int(image_color):
        return False
    if _orbit_keys(source_mask) != source_orbit:
        return False
    if _orbit_keys(target_mask) != target_orbit:
        return False
    if np.array_equal(_apply_transform(target_mask, transform), target_mask):
        return False
    if np.array_equal(meta["output"], recolor_only):
        return False
    if not np.array_equal(meta["output"], expected):
        return False
    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    transform = _next_transform()
    distractor_mode = _next_distractor_mode()
    source_color, image_color = [int(c) for c in random.sample(list(Color.NOT_BLACK), 2)]

    source_mask = _make_asymmetric_mask()
    if source_mask is None:
        return None
    source_orbit = _orbit_keys(source_mask)

    target_kwargs: dict[str, object] = {}
    if distractor_mode == "same_area":
        target_kwargs["exact_area"] = int(source_mask.sum())
    elif distractor_mode == "same_bbox":
        target_kwargs["exact_shape"] = tuple(map(int, source_mask.shape))

    target_mask = _make_asymmetric_mask(forbidden_orbit=source_orbit, **target_kwargs)
    if target_mask is None:
        return None
    target_orbit = _orbit_keys(target_mask)

    image_mask = _apply_transform(source_mask, transform)
    out_mask = _apply_transform(target_mask, transform)
    if np.array_equal(out_mask, target_mask):
        return None

    expected = _render_patch(out_mask, image_color)
    if _output_signature(expected) in _RECENT_OUTPUT_SIGS:
        return None

    rows = random.randint(*GRID_ROWS)
    cols = random.randint(*GRID_COLS)
    canvas = np.full((rows, cols), BG, dtype=int)
    patches = {
        "source": _render_patch(source_mask, source_color),
        "image": _render_patch(image_mask, image_color),
        "target": _render_patch(target_mask, source_color),
    }

    for name in sorted(patches, key=lambda key: patches[key].size, reverse=True):
        try:
            r0, c0 = random_free_location(
                canvas,
                patches[name],
                transparent=BG,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, patches[name], r0, c0, transparent=BG)

    if not _validate_generated(
        canvas,
        transform=transform,
        source_color=source_color,
        image_color=image_color,
        source_orbit=source_orbit,
        target_orbit=target_orbit,
        expected=expected,
    ):
        return None

    _remember_output(_output_signature(expected))
    return canvas, expected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the hidden demo pair of mirror/rotation-equivalent sprites and apply that transform to the third sprite.

    Input structure:
    - A black background contains exactly three separated monochrome sprites.
    - Exactly two sprites are the same shape up to one non-trivial dihedral transform, and the source sprite's color is reused by the third sprite.
    - The demo shape has no non-trivial self-symmetry, so the source-to-image transform is unique.

    Rule:
    - Compare the three sprites pairwise and locate the unique isomorphic pair.
    - Within that pair, the sprite whose color also appears on the third sprite is the source; the other is the transformed image.
    - Apply the same transform to the third sprite, recolor it to the image color, and return the tight crop of that result on black.

    Key insight:
    - The input hides a miniature example input-output pair; positions are irrelevant, and only the repeated source color fixes the transform direction.
    """
    return _analyze_scene(grid)["output"]


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        built = _generate_case()
        if built is None:
            continue
        grid, _expected = built
        return grid
    raise RuntimeError("failed to generate a valid who-mirrors-whom ARC input")
