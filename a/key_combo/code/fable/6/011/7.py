from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_HELPER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import find_components, get_bg, paste


ARC_COLORS = list(range(10))
MAX_GENERATION_ATTEMPTS = 500


def _component_sprite(comp: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    coords = np.asarray(comp["coords"], dtype=int).copy()
    coords[:, 0] -= r0
    coords[:, 1] -= c0
    sprite[coords[:, 0], coords[:, 1]] = int(comp["color"])
    return sprite


def _is_hollow_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(comp["area"]) != 2 * h + 2 * w - 4:
        return False

    mask = np.zeros((h, w), dtype=bool)
    coords = np.asarray(comp["coords"], dtype=int).copy()
    coords[:, 0] -= r0
    coords[:, 1] -= c0
    mask[coords[:, 0], coords[:, 1]] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(mask[border].all() and not mask[~border].any())


def _is_solid_block(comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return int(comp["area"]) == h * w


def _strictly_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    ir0, ic0, ir1, ic1 = map(int, inner)
    or0, oc0, or1, oc1 = map(int, outer)
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _extract_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    frame_comps: list[dict] = []
    solid_comps: list[dict] = []
    for comp in comps:
        if _is_hollow_frame(comp):
            frame_comps.append(comp)
        elif _is_solid_block(comp):
            solid_comps.append(comp)
        else:
            raise ValueError("unexpected object shape in input")

    frame_comps.sort(key=lambda comp: tuple(map(int, comp["bbox"])))
    solid_comps.sort(key=lambda comp: tuple(map(int, comp["bbox"])))

    cargo_owner: dict[int, int] = {}
    frames: list[dict] = []
    for frame_idx, comp in enumerate(frame_comps):
        bbox = tuple(map(int, comp["bbox"]))
        sprite = _component_sprite(comp, bg)
        cargo_idx = None
        for solid_idx, solid in enumerate(solid_comps):
            if _strictly_inside(tuple(map(int, solid["bbox"])), bbox):
                if cargo_idx is not None:
                    raise ValueError("frame contains more than one cargo block")
                cargo_idx = solid_idx
        if cargo_idx is not None:
            if cargo_idx in cargo_owner:
                raise ValueError("cargo block is enclosed by multiple frames")
            cargo_owner[cargo_idx] = frame_idx
            sr0, sc0, sr1, sc1 = map(int, solid_comps[cargo_idx]["bbox"])
            hole_l = sc0 - bbox[1]
            hole_r = sc1 - bbox[1]
            sprite[-1, hole_l : hole_r + 1] = bg
            cargo_bbox = (sr0, sc0, sr1, sc1)
        else:
            cargo_bbox = None

        frames.append(
            {
                "bbox": bbox,
                "sprite": sprite,
                "cargo_bbox": cargo_bbox,
            }
        )

    solids: list[dict] = []
    for solid_idx, comp in enumerate(solid_comps):
        solids.append(
            {
                "bbox": tuple(map(int, comp["bbox"])),
                "sprite": _component_sprite(comp, bg),
                "is_cargo": solid_idx in cargo_owner,
            }
        )
    return bg, frames, solids


def _can_place(out: np.ndarray, sprite: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    h, w = sprite.shape
    if r0 < 0 or c0 < 0 or r0 + h > out.shape[0] or c0 + w > out.shape[1]:
        return False
    mask = sprite != int(bg)
    return not bool(((out[r0 : r0 + h, c0 : c0 + w] != int(bg)) & mask).any())


def _highest_row(out: np.ndarray, sprite: np.ndarray, c0: int, bg: int) -> int:
    limit = out.shape[0] - sprite.shape[0]
    for r0 in range(limit + 1):
        if _can_place(out, sprite, r0, c0, bg):
            return int(r0)
    raise ValueError("frame placement failed")


def _lowest_row(out: np.ndarray, sprite: np.ndarray, c0: int, bg: int) -> int:
    limit = out.shape[0] - sprite.shape[0]
    for r0 in range(limit, -1, -1):
        if _can_place(out, sprite, r0, c0, bg):
            return int(r0)
    raise ValueError("solid placement failed")


def _transform(bg: int, shape: tuple[int, int], frames: list[dict], solids: list[dict]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)

    for obj in sorted(frames, key=lambda item: (item["bbox"][0], item["bbox"][1], item["bbox"][2], item["bbox"][3])):
        c0 = int(obj["bbox"][1])
        r0 = _highest_row(out, obj["sprite"], c0, bg)
        paste(out, obj["sprite"], r0, c0, transparent=bg)

    for obj in sorted(solids, key=lambda item: (-item["bbox"][2], item["bbox"][1], item["bbox"][0], item["bbox"][3])):
        c0 = int(obj["bbox"][1])
        r0 = _lowest_row(out, obj["sprite"], c0, bg)
        paste(out, obj["sprite"], r0, c0, transparent=bg)

    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Split loaded frames into a floating holed frame and a falling cargo block.

    Hollow frames float upward in their original columns and stack tightly.
    Solid blocks, including cargo blocks originally enclosed by frames, sink
    downward in their original columns and stack tightly.
    """
    grid = np.asarray(input_grid, dtype=int)
    if grid.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg, frames, solids = _extract_scene(grid)
    return _transform(bg, grid.shape, frames, solids)


def _frame_sprite(height: int, width: int, color: int, bg: int) -> np.ndarray:
    sprite = np.full((height, width), int(bg), dtype=int)
    sprite[0, :] = int(color)
    sprite[-1, :] = int(color)
    sprite[:, 0] = int(color)
    sprite[:, -1] = int(color)
    return sprite


def _solid_sprite(size: int, color: int) -> np.ndarray:
    return np.full((size, size), int(color), dtype=int)


def _make_empty_frame(width: int, bg: int) -> dict:
    height = random.randint(4, 7)
    color = random.choice([c for c in ARC_COLORS if c != bg])
    sprite = _frame_sprite(height, width, color, bg)
    return {
        "sprite": sprite,
        "outer_height": height,
        "final_height": height,
    }


def _make_loaded_frame(width: int, bg: int) -> dict | None:
    height = random.randint(4, 7)
    inner_h = height - 2
    inner_w = width - 2
    cargo_size = random.randint(1, min(inner_h, inner_w))
    frame_color = random.choice([c for c in ARC_COLORS if c != bg])
    cargo_color = random.choice([c for c in ARC_COLORS if c not in {bg, frame_color}]) if len(ARC_COLORS) > 2 else frame_color

    frame = _frame_sprite(height, width, frame_color, bg)
    cargo = _solid_sprite(cargo_size, cargo_color)
    max_r = height - 1 - cargo_size
    max_c = width - 1 - cargo_size
    if max_r < 1 or max_c < 1:
        return None
    cargo_r = random.randint(1, max_r)
    cargo_c = random.randint(1, max_c)
    paste(frame, cargo, cargo_r, cargo_c, transparent=bg)
    return {
        "sprite": frame,
        "outer_height": height,
        "final_height": height + cargo_size,
    }


def _make_free_solid(width: int, bg: int) -> dict:
    color = random.choice([c for c in ARC_COLORS if c != bg])
    sprite = _solid_sprite(width, color)
    return {
        "sprite": sprite,
        "outer_height": width,
        "final_height": width,
    }


def _build_band(height: int, width: int, bg: int) -> np.ndarray | None:
    n_objects = random.randint(1, 4)
    kind_choices = ["empty", "loaded", "solid"]
    items: list[dict] = []
    while len(items) < n_objects:
        kind = random.choice(kind_choices)
        if kind == "empty":
            item = _make_empty_frame(width, bg)
        elif kind == "loaded":
            item = _make_loaded_frame(width, bg)
        else:
            item = _make_free_solid(width, bg)
        if item is None:
            continue
        items.append(item)

    random.shuffle(items)
    input_total = sum(int(item["outer_height"]) for item in items) + max(0, len(items) - 1)
    final_total = sum(int(item["final_height"]) for item in items)
    if input_total > height or final_total > height:
        return None

    band = np.full((height, width), int(bg), dtype=int)
    extra = height - input_total
    gaps = [0] * (len(items) + 1)
    for idx in range(1, len(items)):
        gaps[idx] = 1
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1

    r0 = gaps[0]
    for idx, item in enumerate(items):
        paste(band, item["sprite"], r0, 0, transparent=bg)
        r0 += int(item["outer_height"]) + gaps[idx + 1]

    return band


def _generate_case() -> np.ndarray | None:
    bg = random.choice(ARC_COLORS)
    height = random.randint(16, 28)
    n_bands = random.randint(1, 3)
    band_widths = [random.randint(3, 6) for _ in range(n_bands)]
    min_width = sum(band_widths) + 2 * max(0, n_bands - 1)
    if min_width > 30:
        return None

    bands: list[np.ndarray] = []
    for width in band_widths:
        band = _build_band(height, width, bg)
        if band is None:
            return None
        bands.append(band)

    width = random.randint(min_width, 30)
    grid = np.full((height, width), int(bg), dtype=int)

    slack = width - min_width
    gaps = [0] * (n_bands + 1)
    for idx in range(1, n_bands):
        gaps[idx] = 2
    for _ in range(slack):
        gaps[random.randrange(len(gaps))] += 1

    c0 = gaps[0]
    for idx, band in enumerate(bands):
        paste(grid, band, 0, c0, transparent=bg)
        c0 += band.shape[1] + gaps[idx + 1]

    if int(np.count_nonzero(grid != bg)) == 0:
        return None
    if int(np.count_nonzero(grid == bg)) <= int(np.count_nonzero(grid != bg)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
