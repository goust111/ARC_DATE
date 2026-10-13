from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS) not in sys.path:
    sys.path.append(str(ARC_UTILS))

from arc_utils import Color, bbox_coords, find_components, neighbors4


BG = Color.BLACK
FRAME = Color.GREY
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE]
GEN_RETRIES = 500
AREA_RANGE = range(3, 11)


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _is_channel(component: dict) -> bool:
    if int(component["color"]) != FRAME:
        return False
    r0, c0, r1, c1 = map(int, component["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 2 or w < 4:
        return False
    expected = {(r1, c) for c in range(c0, c1 + 1)}
    for r in range(r0, r1):
        expected.add((r, c0))
        expected.add((r, c1))
    return _coord_set(component["coords"]) == expected


def _detect_channels(grid: np.ndarray) -> list[dict]:
    channels = [
        comp
        for comp in find_components(grid, bg=BG, connectivity=4, monochromatic=True)
        if int(comp["color"]) == FRAME
    ]
    assert channels, "expected gray channels"
    assert all(_is_channel(comp) for comp in channels), "all gray components must be channels"
    return sorted(channels, key=lambda comp: tuple(map(int, comp["bbox"])))


def _detect_objects(grid: np.ndarray) -> list[dict]:
    objects = [
        comp
        for comp in find_components(grid, bg=BG, connectivity=4, monochromatic=True)
        if int(comp["color"]) != FRAME
    ]
    assert objects, "expected colored objects"
    colors = [int(comp["color"]) for comp in objects]
    assert len(colors) == len(set(colors)), "each object must have a distinct color"
    return sorted(objects, key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))


def _channel_interior(channel: dict) -> tuple[int, int]:
    _r0, c0, _r1, c1 = map(int, channel["bbox"])
    return c0 + 1, c1 - 1


def _match_channels(objects: list[dict], channels: list[dict]) -> list[tuple[dict, dict]]:
    pairs: list[tuple[dict, dict]] = []
    used = set()
    for obj in objects:
        _r0, oc0, _r1, oc1 = map(int, obj["bbox"])
        candidates = []
        for idx, channel in enumerate(channels):
            ic0, ic1 = _channel_interior(channel)
            if ic0 <= oc0 and oc1 <= ic1:
                candidates.append(idx)
        assert len(candidates) == 1, "each object must belong to exactly one channel"
        idx = candidates[0]
        assert idx not in used, "channels must map one-to-one to objects"
        used.add(idx)
        pairs.append((obj, channels[idx]))
    assert len(pairs) == len(channels), "expected one object per channel"
    return pairs


def _drop_objects(grid: np.ndarray, objects: list[dict]) -> tuple[np.ndarray, list[dict]]:
    out = np.asarray(grid, dtype=int).copy()
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    height = int(out.shape[0])
    dropped: list[dict] = []
    for obj in objects:
        color = int(obj["color"])
        r0, c0, r1, c1 = map(int, obj["bbox"])
        delta = height - 1 - r1
        dest = obj["coords"].copy()
        dest[:, 0] += delta
        assert dest[:, 0].max() == height - 1
        assert np.all(out[dest[:, 0], dest[:, 1]] == BG), "dropped objects must land on background"
        out[dest[:, 0], dest[:, 1]] = color
        dropped.append(
            {
                "color": color,
                "coords": dest,
                "bbox": bbox_coords(dest),
                "area": int(dest.shape[0]),
            }
        )
    return out, dropped


def _growth_cells(grid: np.ndarray, obj: dict) -> set[tuple[int, int]]:
    height, width = grid.shape
    add: set[tuple[int, int]] = set()
    for r, c in obj["coords"]:
        r = int(r)
        c = int(c)
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < height and 0 <= nc < width and int(grid[nr, nc]) == BG:
                add.add((int(nr), int(nc)))
    return add


def _shrunken_cells(grid: np.ndarray, obj: dict) -> set[tuple[int, int]]:
    height, width = grid.shape
    keep: set[tuple[int, int]] = set()
    for r, c in obj["coords"]:
        r = int(r)
        c = int(c)
        if r in (0, height - 1) or c in (0, width - 1):
            continue
        removable = False
        for nr, nc in neighbors4(r, c):
            if int(grid[nr, nc]) == BG:
                removable = True
                break
        if not removable:
            keep.add((r, c))
    return keep


def _transform_from_objects(grid: np.ndarray, dropped_grid: np.ndarray, dropped_objects: list[dict]) -> np.ndarray:
    areas = [int(obj["area"]) for obj in dropped_objects]
    assert len(areas) >= 3, "expected at least three objects"
    assert len(areas) == len(set(areas)), "object areas must be pairwise distinct"
    max_area = max(areas)
    min_area = min(areas)
    winner = next(obj for obj in dropped_objects if int(obj["area"]) == max_area)
    loser = next(obj for obj in dropped_objects if int(obj["area"]) == min_area)

    grow = _growth_cells(dropped_grid, winner)
    keep = _shrunken_cells(dropped_grid, loser)

    out = dropped_grid.copy()
    loser_coords = loser["coords"]
    out[loser_coords[:, 0], loser_coords[:, 1]] = BG
    for r, c in keep:
        out[r, c] = int(loser["color"])
    for r, c in grow:
        out[r, c] = int(winner["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each colored object through its gray bracket lane, then let the unique largest object grow and the unique smallest object shrink.

    Input structure:
    - The background is black.
    - Several gray 4-connected bracket channels sit above the bottom edge; each channel is an upward-open U made of two vertical arms and one bottom bar.
    - Exactly one monochrome 4-connected colored object floats above each channel, with its column span strictly inside that channel interior.

    Rule:
    - Identify the gray channels and the colored objects, pairing each object to the unique channel whose interior columns contain the object's bounding-box columns.
    - Remove every colored object from its original location and drop it straight downward without changing shape or columns until its lowest pixel reaches the grid bottom.
    - After all objects have landed, compare their areas: the unique largest object grows by one 4-neighbor layer into black cells only, the unique smallest object simultaneously loses every pixel that touches black in the 4-neighborhood or lies on the grid boundary, and every middle-ranked object stays unchanged.
    - Gray channel pixels remain unchanged throughout.

    Key insight:
    - The bracket channels determine which object falls in which vertical lane, and the post-fall neighborhood created by that landing is exactly what the later area-ranked growth and shrink operations must use.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    channels = _detect_channels(g)
    objects = _detect_objects(g)
    _match_channels(objects, channels)
    dropped_grid, dropped_objects = _drop_objects(g, objects)
    return _transform_from_objects(g, dropped_grid, dropped_objects)


def _relative_eroded_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    eroded = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if r == h - 1 or r == 0 or c == 0 or c == w - 1:
                continue
            if mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]:
                eroded[r, c] = True
    return eroded


def _connected_mask(area: int, max_width: int, rng: random.Random, survivor: bool | None = None) -> np.ndarray | None:
    options = [
        (h, w)
        for h in range(2, 7)
        for w in range(1, min(max_width, 5) + 1)
        if h * w >= area
    ]
    rng.shuffle(options)
    for h, w in options:
        for _ in range(80):
            start = (rng.randrange(h), rng.randrange(w))
            cells = {start}
            while len(cells) < area:
                frontier = []
                for r, c in tuple(cells):
                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                            frontier.append((nr, nc))
                if not frontier:
                    break
                cells.add(rng.choice(frontier))
            if len(cells) != area:
                continue

            rows = sorted({r for r, _c in cells})
            cols = sorted({c for _r, c in cells})
            r0, r1 = rows[0], rows[-1]
            c0, c1 = cols[0], cols[-1]
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in cells:
                mask[r - r0, c - c0] = True
            if mask.shape[1] > max_width:
                continue
            if mask.shape[1] >= max_width + 1:
                continue
            if mask.shape[0] == 1 and area > 4:
                continue
            if area >= 5 and bool(mask.all()) and rng.random() < 0.75:
                continue
            eroded = _relative_eroded_mask(mask)
            survives = bool(eroded.any())
            if survivor is not None and survives != survivor:
                continue
            return mask
    return None


def _paint_channel(grid: np.ndarray, r_bottom: int, c0: int, width: int, height: int) -> None:
    c1 = c0 + width - 1
    grid[r_bottom, c0 : c1 + 1] = FRAME
    for r in range(r_bottom - height + 1, r_bottom):
        grid[r, c0] = FRAME
        grid[r, c1] = FRAME


def _place_shape(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    view = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    assert np.all(view[mask] == BG)
    view[mask] = int(color)


def _choose_areas(rng: random.Random) -> tuple[list[int], bool]:
    if rng.random() < 0.5:
        return [8, 9, 10], True

    low = rng.sample(list(range(3, 7)), 2)
    high = rng.choice([8, 9, 10])
    areas = sorted(low + [high])
    return areas, False


def _layout(rng: random.Random) -> tuple[int, int, list[tuple[int, int, int, int]]] | None:
    widths = rng.sample([5, 6, 7, 8], 3)
    widths.sort()
    gaps = [rng.randint(3, 5) for _ in range(2)]
    used_w = sum(widths) + sum(gaps)
    if used_w > 28:
        return None

    height = rng.randint(19, 27)
    width = rng.randint(used_w + 1, 30)
    left_margin = rng.randint(0, width - used_w)
    r_bottom = height - rng.randint(4, 6)

    channels = []
    c = left_margin
    for ch_width in widths:
        ch_height = rng.randint(3, 6)
        if r_bottom - ch_height < 2:
            return None
        channels.append((r_bottom, c, ch_width, ch_height))
        c += ch_width
        if len(channels) < 3:
            c += gaps[len(channels) - 1]
    return height, width, channels


def _build_instance(rng: random.Random) -> np.ndarray | None:
    laid_out = _layout(rng)
    if laid_out is None:
        return None
    height, width, channels = laid_out
    areas, smallest_survives = _choose_areas(rng)

    grid = np.zeros((height, width), dtype=int)
    for r_bottom, c0, ch_width, ch_height in channels:
        _paint_channel(grid, r_bottom, c0, ch_width, ch_height)

    colors = rng.sample(PALETTE, 3)
    channel_order = list(range(3))
    rng.shuffle(channel_order)
    assigned_areas = areas[:]
    rng.shuffle(assigned_areas)

    for rank, idx in enumerate(channel_order):
        r_bottom, c0, ch_width, ch_height = channels[idx]
        interior_width = ch_width - 2
        area = assigned_areas[rank]
        is_smallest = area == min(areas)
        want_survivor = smallest_survives if is_smallest else None
        mask = _connected_mask(area, interior_width - 1, rng, survivor=want_survivor)
        if mask is None:
            return None

        max_top = r_bottom - ch_height - mask.shape[0] - 1
        if max_top < 1:
            return None
        top = rng.randint(1, max_top)
        spare = interior_width - mask.shape[1]
        if spare <= 0:
            return None
        left = c0 + 1 + rng.randint(0, spare - 1)
        _place_shape(grid, mask, top, left, colors[idx])

    if not _validate_instance(grid, smallest_survives):
        return None
    return grid


def _validate_instance(inp: np.ndarray, smallest_survives: bool) -> bool:
    try:
        channels = _detect_channels(inp)
        objects = _detect_objects(inp)
        pairs = _match_channels(objects, channels)
        if len(channels) != 3 or len(objects) != 3:
            return False
        areas = sorted(int(obj["area"]) for obj in objects)
        if len(set(areas)) != 3:
            return False
        if smallest_survives and areas != [8, 9, 10]:
            return False
        if not smallest_survives and areas[0] >= 7:
            return False

        for obj, channel in pairs:
            ic0, ic1 = _channel_interior(channel)
            _r0, oc0, _r1, oc1 = map(int, obj["bbox"])
            if not (ic0 <= oc0 and oc1 <= ic1 and (oc1 - oc0) < (ic1 - ic0 + 1)):
                return False

        dropped_grid, dropped_objects = _drop_objects(inp, objects)
        transformed = _transform_from_objects(inp, dropped_grid, dropped_objects)
        if np.array_equal(transformed, dropped_grid):
            return False

        areas_after_drop = [int(obj["area"]) for obj in dropped_objects]
        winner = dropped_objects[areas_after_drop.index(max(areas_after_drop))]
        loser = dropped_objects[areas_after_drop.index(min(areas_after_drop))]
        grow = _growth_cells(dropped_grid, winner)
        keep = _shrunken_cells(dropped_grid, loser)
        if not grow:
            return False
        if len(keep) == int(loser["area"]):
            return False
        if smallest_survives != bool(keep):
            return False

        solved = main(inp)
        if not np.array_equal(solved, transformed):
            return False

        just_drop = dropped_grid
        just_growth = dropped_grid.copy()
        for r, c in grow:
            just_growth[r, c] = int(winner["color"])
        just_shrink = dropped_grid.copy()
        loser_coords = loser["coords"]
        just_shrink[loser_coords[:, 0], loser_coords[:, 1]] = BG
        for r, c in keep:
            just_shrink[r, c] = int(loser["color"])
        if np.array_equal(transformed, just_drop):
            return False
        if np.array_equal(transformed, just_growth):
            return False
        if np.array_equal(transformed, just_shrink):
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _build_instance(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-growth-shrink ARC instance")
