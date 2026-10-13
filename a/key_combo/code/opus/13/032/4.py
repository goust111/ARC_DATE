from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SPRITE_COLORS = (1, 2, 3)
SPRITE_SIZE = 3
ZONE_SIZE = 5
SLOT_STEP = 6
GEN_ATTEMPTS = 400
SHAPE_AREA_MIN = 4
SHAPE_AREA_MAX = 6

_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_SLOT_COORDS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = set()
    stack = [tuple(map(int, coords[0]))]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for dr, dc in _DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
    monochromatic: bool = True,
) -> list[dict]:
    if connectivity != 4:
        raise ValueError("this task uses only 4-connectivity")
    if not monochromatic:
        raise ValueError("this task uses only monochromatic components")

    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in _DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _outer_corner_positions(bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {
        "tl": (r0 - 1, c0 - 1),
        "tr": (r0 - 1, c1 + 1),
        "bl": (r1 + 1, c0 - 1),
        "br": (r1 + 1, c1 + 1),
    }


def _rotation_k(direction: str) -> int:
    if direction == "cw":
        return 3
    if direction == "ccw":
        return 1
    raise ValueError(f"unknown direction {direction}")


def _rot_asymmetric(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k=k)) for k in (1, 2, 3))


def _shape_catalog() -> list[np.ndarray]:
    classes: dict[tuple[int, ...], np.ndarray] = {}
    for bits in range(1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        area = int(mask.sum())
        if area < SHAPE_AREA_MIN or area > SHAPE_AREA_MAX:
            continue
        if bbox_coords(np.argwhere(mask)) != (0, 0, 2, 2):
            continue
        if not _connected4(mask):
            continue
        if not _rot_asymmetric(mask):
            continue
        canon = min(tuple(np.rot90(mask, k=k).astype(int).ravel()) for k in range(4))
        classes.setdefault(canon, np.array(canon, dtype=bool).reshape(3, 3))
    shapes = list(classes.values())
    if len(shapes) < 3:
        raise RuntimeError("insufficient valid 3x3 sprite shapes")
    return shapes


_VALID_SHAPES = _shape_catalog()


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    bg = get_bg(g)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    sprites = [comp for comp in comps if int(comp["area"]) > 1]
    markers = [comp for comp in comps if int(comp["area"]) == 1]
    assert len(sprites) == 3, "expected exactly three sprite components"
    assert len(markers) == 6, "expected exactly six singleton markers"

    sprite_by_color: dict[int, dict] = {}
    for sprite in sprites:
        color = int(sprite["color"])
        assert color not in sprite_by_color, "sprite colors must be unique"
        r0, c0, r1, c1 = sprite["bbox"]
        assert (r1 - r0 + 1, c1 - c0 + 1) == (SPRITE_SIZE, SPRITE_SIZE), "sprite bbox must be 3x3"
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert set(map(int, np.unique(patch))) <= {bg, color}, "sprite bbox must contain only bg and its body color"
        mask = patch == color
        assert bbox_coords(np.argwhere(mask)) == (0, 0, 2, 2), "sprite mask must span the full 3x3 bbox"
        assert _connected4(mask), "sprite must be 4-connected"
        assert _rot_asymmetric(mask), "sprite must be asymmetric under every quarter-turn"
        sprite_by_color[color] = {
            "color": color,
            "bbox": sprite["bbox"],
            "coords": sprite["coords"],
            "patch": patch,
            "mask": mask,
            "center": (r0 + 1, c0 + 1),
        }

    marker_lookup: dict[tuple[int, int], int] = {}
    marker_colors: list[int] = []
    for marker in markers:
        (r, c) = map(int, marker["coords"][0])
        marker_lookup[(r, c)] = int(marker["color"])
        marker_colors.append(int(marker["color"]))
    assert set(sprite_by_color) == set(marker_colors), "marker colors must match the three sprite colors"
    for color in sprite_by_color:
        assert marker_colors.count(color) == 2, "each color must appear on exactly one marker pair"

    sprite_infos = []
    used_markers = set()
    color_map: dict[int, int] = {}
    for color, info in sprite_by_color.items():
        outer = _outer_corner_positions(info["bbox"])
        outer_hits = {name: marker_lookup[pos] for name, pos in outer.items() if pos in marker_lookup}
        assert len(outer_hits) == 2, "each sprite must have exactly two outer-corner markers"

        direction = None
        marker_color = None
        if set(outer_hits) == {"tl", "br"} and outer_hits["tl"] == outer_hits["br"]:
            direction = "cw"
            marker_color = int(outer_hits["tl"])
            used_markers |= {outer["tl"], outer["br"]}
        if set(outer_hits) == {"tr", "bl"} and outer_hits["tr"] == outer_hits["bl"]:
            assert direction is None, "marker pair must use exactly one diagonal"
            direction = "ccw"
            marker_color = int(outer_hits["tr"])
            used_markers |= {outer["tr"], outer["bl"]}
        assert direction is not None, "markers must occupy one full outside diagonal"
        assert marker_color != color, "marker color must point to a different sprite"
        color_map[color] = marker_color
        sprite_infos.append(
            {
                **info,
                "direction": direction,
                "marker_color": marker_color,
                "target_center": sprite_by_color[marker_color]["center"],
            }
        )

    assert used_markers == set(marker_lookup), "every singleton marker must belong to exactly one sprite"
    assert len(set(color_map.values())) == 3, "target mapping must be a permutation"
    for color in color_map:
        assert color_map[color] != color
        assert color_map[color_map[color]] != color
        assert color_map[color_map[color_map[color]]] == color, "marker-color mapping must form a 3-cycle"

    return {
        "grid": g,
        "bg": bg,
        "sprite_infos": sprite_infos,
        "sprite_by_color": sprite_by_color,
        "marker_lookup": marker_lookup,
        "color_map": color_map,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each 3x3 sprite by the direction encoded by its diagonal marker pair, then move it to the sprite whose color matches that pair.

    Input structure:
    - The grid has a black background with exactly three small monochrome sprites and six singleton markers.
    - Each sprite is a 4-connected asymmetric shape whose bounding box is 3x3, and the three sprite body colors are distinct.
    - Around each sprite, exactly two same-color markers sit one cell outside opposite bbox corners on either the TL-BR diagonal or the TR-BL diagonal.
    - The color of a marker pair matches the body color of a different sprite, so the three marker-color links form one 3-cycle.

    Rule:
    - For each sprite, read its two outside-corner markers and identify which diagonal they occupy.
    - TL plus BR means rotate that sprite 90 degrees clockwise; TR plus BL means rotate it 90 degrees counterclockwise.
    - The marker color names the destination sprite, so move the rotated sprite to the original center of the sprite with that body color.
    - Compute all rotations and destinations from the original input, perform the moves simultaneously, and erase all marker pixels.

    Key insight:
    - The same marker pair carries two independent signals: diagonal placement chooses the rotation direction, while marker color chooses the destination center.
    """
    analysis = _analyze(grid)
    g = analysis["grid"]
    bg = analysis["bg"]

    out = g.copy()
    for pos in analysis["marker_lookup"]:
        out[pos] = bg
    for info in analysis["sprite_infos"]:
        for r, c in info["coords"]:
            out[int(r), int(c)] = bg

    for info in analysis["sprite_infos"]:
        rotated = np.rot90(info["patch"], k=_rotation_k(info["direction"]))
        tr, tc = info["target_center"]
        r0 = tr - 1
        c0 = tc - 1
        window = out[r0 : r0 + SPRITE_SIZE, c0 : c0 + SPRITE_SIZE]
        assert window.shape == (SPRITE_SIZE, SPRITE_SIZE), "translated sprite must remain in bounds"
        assert not np.any(window != bg), "simultaneous moves must land on cleared cells"
        paste(out, rotated, r0, c0, transparent=bg)

    return out


def _marker_positions(bbox: tuple[int, int, int, int], direction: str) -> list[tuple[int, int]]:
    outer = _outer_corner_positions(bbox)
    if direction == "cw":
        return [outer["tl"], outer["br"]]
    if direction == "ccw":
        return [outer["tr"], outer["bl"]]
    raise ValueError(f"unknown direction {direction}")


def _cycle_map(colors: list[int], forward: bool) -> dict[int, int]:
    step = 1 if forward else -1
    n = len(colors)
    return {colors[i]: colors[(i + step) % n] for i in range(n)}


def _render_instance(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        zr, zc = spec["zone"]
        r0 = zr + 1
        c0 = zc + 1
        patch = spec["patch"]
        paste(grid, patch, r0, c0, transparent=BG)
        bbox = (r0, c0, r0 + 2, c0 + 2)
        for mr, mc in _marker_positions(bbox, spec["direction"]):
            assert int(grid[mr, mc]) == BG
            grid[mr, mc] = int(spec["marker_color"])
    return grid


def _validate_generated_instance(grid: np.ndarray) -> None:
    analysis = _analyze(grid)
    directions = {info["direction"] for info in analysis["sprite_infos"]}
    assert directions == {"cw", "ccw"}, "generator must expose both rotation branches"

    out = main(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid), "input and output should not collapse to the same grid"
    for pos in analysis["marker_lookup"]:
        assert int(out[pos]) == analysis["bg"], "all markers must be erased"

    moved_centers = {info["target_center"] for info in analysis["sprite_infos"]}
    assert len(moved_centers) == 3
    for info in analysis["sprite_infos"]:
        tr, tc = info["target_center"]
        expected = np.rot90(info["patch"], k=_rotation_k(info["direction"]))
        actual = out[tr - 1 : tr + 2, tc - 1 : tc + 2]
        assert np.array_equal(actual, expected), "output patch must equal the correctly rotated source sprite"

    out_comps = find_components(out, bg=analysis["bg"], connectivity=4, monochromatic=True)
    assert len(out_comps) == 3, "output must contain only the three moved sprites"
    assert all(int(comp["area"]) > 1 for comp in out_comps), "output must not contain singleton markers"
    assert {int(comp["color"]) for comp in out_comps} == set(SPRITE_COLORS)


def _gen_attempt() -> np.ndarray | None:
    top_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    height = top_margin + 11 + bottom_margin
    width = left_margin + 11 + right_margin

    omit = random.randrange(4)
    chosen_slots = [slot for slot in range(4) if slot != omit]
    random.shuffle(chosen_slots)

    colors = list(random.sample(SPRITE_COLORS, 3))
    color_map = _cycle_map(colors, forward=random.choice([True, False]))

    directions = ["cw", "ccw", random.choice(["cw", "ccw"])]
    random.shuffle(directions)

    patch_indices = random.sample(range(len(_VALID_SHAPES)), 3)
    specs = []
    for slot, color, direction, patch_index in zip(chosen_slots, colors, directions, patch_indices):
        sr, sc = _SLOT_COORDS[slot]
        zone = (top_margin + sr * SLOT_STEP, left_margin + sc * SLOT_STEP)
        patch = np.full((SPRITE_SIZE, SPRITE_SIZE), BG, dtype=int)
        mask = _VALID_SHAPES[patch_index].copy()
        mask = np.rot90(mask, k=random.randrange(4))
        patch[mask] = color
        specs.append(
            {
                "zone": zone,
                "patch": patch,
                "direction": direction,
                "marker_color": color_map[color],
            }
        )

    try:
        grid = _render_instance(specs, (height, width))
        _validate_generated_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
