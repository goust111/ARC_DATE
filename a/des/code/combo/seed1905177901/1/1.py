from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _bg_color(g: np.ndarray) -> int:
    if bool(np.any(g == BG)):
        return BG
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[int(cnts.argmax())])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _ in coords]
    cs = [int(c) for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=bool)


def _mask_from_coords(coords: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _canonical_shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    base = _crop_mask(np.asarray(mask, dtype=bool))
    best = None
    for k in range(4):
        rot = np.rot90(base, k)
        for variant in (rot, np.fliplr(rot)):
            cropped = _crop_mask(variant)
            key = (int(cropped.shape[0]), int(cropped.shape[1]), cropped.tobytes())
            if best is None or key < best:
                best = key
    assert best is not None
    return best


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _monochrome_components(g: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == int(bg) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not bool(seen[nr, nc])
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            r0, c0, r1, c1 = _bbox(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[int(rr) - r0, int(cc) - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                }
            )
    return comps


def _extract_holes(g: np.ndarray, comp: dict, bg: int) -> list[dict] | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    sub = np.asarray(g[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    color = int(comp["color"])
    if sub.shape[0] < 4 or sub.shape[1] < 4:
        return None
    if set(map(int, np.unique(sub))) - {int(bg), color}:
        return None
    if not (
        bool(np.all(sub[0, :] == color))
        and bool(np.all(sub[-1, :] == color))
        and bool(np.all(sub[:, 0] == color))
        and bool(np.all(sub[:, -1] == color))
    ):
        return None

    holes: list[dict] = []
    for coords in _components_from_mask(sub == int(bg)):
        if any(
            rr == 0 or cc == 0 or rr == sub.shape[0] - 1 or cc == sub.shape[1] - 1
            for rr, cc in coords
        ):
            continue
        hole_mask = _mask_from_coords(coords, sub.shape)
        hole_crop = _crop_mask(hole_mask)
        holes.append(
            {
                "area": int(len(coords)),
                "shape_key": _canonical_shape_key(hole_crop),
            }
        )
    return holes or None


def _parse_scene(g: np.ndarray) -> tuple[int, list[dict], dict[tuple[int, int, bytes], dict], list[dict]]:
    bg = _bg_color(g)
    comps = _monochrome_components(g, bg)
    frames: list[dict] = []
    sprites: list[dict] = []
    sprite_by_shape: dict[tuple[int, int, bytes], dict] = {}

    for comp in comps:
        holes = _extract_holes(g, comp, bg)
        if holes:
            frames.append(
                {
                    "bbox": comp["bbox"],
                    "color": int(comp["color"]),
                    "holes": holes,
                }
            )
            continue

        key = _canonical_shape_key(comp["mask"])
        assert key not in sprite_by_shape, "sprite shapes must be globally unique"
        sprite = {
            "bbox": comp["bbox"],
            "color": int(comp["color"]),
            "shape_key": key,
            "mask": np.asarray(comp["mask"], dtype=bool),
        }
        sprite_by_shape[key] = sprite
        sprites.append(sprite)

    assert frames, "at least one frame is required"
    return bg, frames, sprite_by_shape, sprites


def _paint_border(out: np.ndarray, bbox: tuple[int, int, int, int], colors: list[int]) -> None:
    assert colors
    edge_colors = [int(colors[i % len(colors)]) for i in range(4)]
    if len(colors) > 4:
        edge_colors = [int(colors[i]) for i in range(4)]
    r0, c0, r1, c1 = map(int, bbox)
    out[r0, c0 : c1 + 1] = edge_colors[0]
    out[r0 + 1 : r1 + 1, c1] = edge_colors[1]
    out[r1, c0:c1] = edge_colors[2]
    out[r0 + 1 : r1, c0] = edge_colors[3]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each enclosed background hole to an outside sprite by shape, sort the
    matched colors by hole area, recolor the frame border clockwise, and erase
    all outside sprites.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames, sprite_by_shape, sprites = _parse_scene(g)

    out = g.copy()
    for sprite in sprites:
        r0, c0, r1, c1 = map(int, sprite["bbox"])
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[np.asarray(sprite["mask"], dtype=bool)] = int(bg)

    for frame in frames:
        matched: list[tuple[int, int]] = []
        for hole in frame["holes"]:
            key = hole["shape_key"]
            assert key in sprite_by_shape, "every hole must have a matching sprite"
            matched.append((int(hole["area"]), int(sprite_by_shape[key]["color"])))
        matched.sort(key=lambda item: item[0])
        frame_colors = [int(color) for _area, color in matched]
        _paint_border(out, frame["bbox"], frame_colors)

    return out


def _ascii_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


_BASE_SHAPES = {
    1: [_ascii_mask(["#"])],
    2: [_ascii_mask(["##"])],
    3: [_ascii_mask(["###"]), _ascii_mask(["##", "#."])],
    4: [
        _ascii_mask(["####"]),
        _ascii_mask(["##", "##"]),
        _ascii_mask(["###", ".#."]),
        _ascii_mask(["###", "#.."]),
        _ascii_mask([".##", "##."]),
    ],
    5: [
        _ascii_mask(["#####"]),
        _ascii_mask([".#.", "###", ".#."]),
        _ascii_mask(["##", "##", "#."]),
        _ascii_mask(["####", "#..."]),
        _ascii_mask(["###", "#.#"]),
    ],
    6: [
        _ascii_mask(["######"]),
        _ascii_mask(["###", "###"]),
        _ascii_mask(["####", "##.."]),
        _ascii_mask(["###", "##.", "#.."]),
        _ascii_mask(["##.", "###", ".#."]),
    ],
}


def _validate_shape_catalog() -> None:
    for area, masks in _BASE_SHAPES.items():
        seen = set()
        for mask in masks:
            assert int(mask.sum()) == int(area)
            key = _canonical_shape_key(mask)
            assert key not in seen
            seen.add(key)


_validate_shape_catalog()


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(np.asarray(mask, dtype=bool), random.randrange(4))
    if random.choice([False, True]):
        out = np.fliplr(out)
    return np.asarray(out, dtype=bool)


def _sample_primary_areas(mode: str) -> list[int]:
    if mode == "few":
        return sorted(random.sample([3, 4, 5, 6], 3))
    if mode == "exact4":
        return sorted(random.sample([1, 3, 4, 5, 6], 4))
    assert mode == "many"
    count = random.choice([5, 6])
    return sorted(random.sample([1, 2, 3, 4, 5, 6], count))


def _sample_secondary_areas(primary_areas: list[int]) -> list[int]:
    repeatable = [area for area in primary_areas if len(_BASE_SHAPES[area]) >= 2]
    assert len(repeatable) >= 2
    return sorted(random.sample(repeatable, 2))


def _choose_shape_masks() -> tuple[list[dict], list[dict]]:
    mode = random.choice(["few", "exact4", "many"])
    primary_areas = _sample_primary_areas(mode)
    secondary_areas = _sample_secondary_areas(primary_areas)

    used_variant: dict[int, set[int]] = {area: set() for area in _BASE_SHAPES}

    def build_holes(area_list: list[int]) -> list[dict]:
        holes: list[dict] = []
        for area in area_list:
            choices = [i for i in range(len(_BASE_SHAPES[area])) if i not in used_variant[area]]
            assert choices, "ran out of unique shape variants for a repeated area"
            idx = random.choice(choices)
            used_variant[area].add(int(idx))
            base_mask = np.asarray(_BASE_SHAPES[area][idx], dtype=bool)
            holes.append(
                {
                    "area": int(area),
                    "shape_key": _canonical_shape_key(base_mask),
                    "hole_mask": _random_transform(base_mask),
                    "sprite_mask": _random_transform(base_mask),
                }
            )
        return holes

    return build_holes(primary_areas), build_holes(secondary_areas)


def _can_place_hole(occupied: np.ndarray, hole_mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = map(int, hole_mask.shape)
    for rr, cc in np.argwhere(hole_mask):
        gr = int(r0 + rr)
        gc = int(c0 + cc)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < occupied.shape[0] and 0 <= nc < occupied.shape[1] and bool(occupied[nr, nc]):
                    return False
    return True


def _build_frame(frame_color: int, holes: list[dict], sprite_colors: list[int]) -> dict | None:
    assert len(holes) == len(sprite_colors)
    count = len(holes)
    largest_h = max(int(h["hole_mask"].shape[0]) for h in holes)
    largest_w = max(int(h["hole_mask"].shape[1]) for h in holes)

    for _ in range(120):
        frame_h = random.randint(max(8, largest_h + 4, count * 2 + 3), max(12, largest_h + count * 3 + 4))
        frame_w = random.randint(max(9, largest_w + 5, count * 3 + 4), max(15, largest_w + count * 4 + 5))
        patch = np.full((frame_h, frame_w), int(frame_color), dtype=int)
        occupied = np.zeros((frame_h, frame_w), dtype=bool)
        placements: list[tuple[int, int] | None] = [None] * count
        order = sorted(range(count), key=lambda i: int(holes[i]["area"]), reverse=True)
        ok = True

        for idx in order:
            hole_mask = np.asarray(holes[idx]["hole_mask"], dtype=bool)
            hh, hw = map(int, hole_mask.shape)
            choices: list[tuple[int, int]] = []
            for r0 in range(1, frame_h - hh):
                for c0 in range(1, frame_w - hw):
                    if _can_place_hole(occupied, hole_mask, r0, c0):
                        choices.append((int(r0), int(c0)))
            if not choices:
                ok = False
                break
            r0, c0 = random.choice(choices)
            placements[idx] = (int(r0), int(c0))
            window = patch[r0 : r0 + hh, c0 : c0 + hw]
            window[hole_mask] = BG
            occ_window = occupied[r0 : r0 + hh, c0 : c0 + hw]
            occ_window[hole_mask] = True

        if not ok:
            continue

        color_order = [
            int(color)
            for _area, color in sorted(
                [(int(holes[i]["area"]), int(sprite_colors[i])) for i in range(count)],
                key=lambda item: item[0],
            )
        ]
        return {
            "patch": patch,
            "bbox_shape": patch.shape,
            "ordered_colors": color_order,
            "sprite_specs": [
                {
                    "shape_key": holes[i]["shape_key"],
                    "mask": np.asarray(holes[i]["sprite_mask"], dtype=bool),
                    "color": int(sprite_colors[i]),
                }
                for i in range(count)
            ],
        }
    return None


def _place_rectangles(shapes: list[tuple[int, int]], h: int, w: int, gap: int = 1) -> list[tuple[int, int]] | None:
    reserved = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(shapes)
    order = sorted(range(len(shapes)), key=lambda i: shapes[i][0] * shapes[i][1], reverse=True)

    for idx in order:
        oh, ow = map(int, shapes[idx])
        choices: list[tuple[int, int]] = []
        for r0 in range(1, h - oh):
            for c0 in range(1, w - ow):
                rr0 = max(0, r0 - gap)
                rr1 = min(h, r0 + oh + gap)
                cc0 = max(0, c0 - gap)
                cc1 = min(w, c0 + ow + gap)
                if not bool(reserved[rr0:rr1, cc0:cc1].any()):
                    choices.append((int(r0), int(c0)))
        if not choices:
            return None
        r0, c0 = random.choice(choices)
        positions[idx] = (int(r0), int(c0))
        rr0 = max(0, r0 - gap)
        rr1 = min(h, r0 + oh + gap)
        cc0 = max(0, c0 - gap)
        cc1 = min(w, c0 + ow + gap)
        reserved[rr0:rr1, cc0:cc1] = True

    return [(int(r), int(c)) for r, c in positions]


def _make_sprite_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(color)
    return patch


def _paste_nonzero(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = map(int, patch.shape)
    window = dst[r0 : r0 + h, c0 : c0 + w]
    window[patch != BG] = patch[patch != BG]


def _expected_output(inp: np.ndarray, frame_infos: list[dict], sprite_infos: list[dict]) -> np.ndarray:
    out = np.asarray(inp, dtype=int).copy()
    for sprite in sprite_infos:
        r0, c0, r1, c1 = map(int, sprite["bbox"])
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[np.asarray(sprite["mask"], dtype=bool)] = BG
    for frame in frame_infos:
        _paint_border(out, frame["bbox"], frame["ordered_colors"])
    return out


def _gen_attempt() -> np.ndarray | None:
    frame_colors = random.sample(COLORS, 2)
    primary_holes, secondary_holes = _choose_shape_masks()

    sprite_palette = [color for color in COLORS if color not in frame_colors]
    primary_sprite_colors = random.sample(sprite_palette, len(primary_holes))
    secondary_sprite_colors = random.sample(sprite_palette, len(secondary_holes))

    primary_frame = _build_frame(int(frame_colors[0]), primary_holes, primary_sprite_colors)
    secondary_frame = _build_frame(int(frame_colors[1]), secondary_holes, secondary_sprite_colors)
    if primary_frame is None or secondary_frame is None:
        return None

    frames = [primary_frame, secondary_frame]
    sprite_specs = primary_frame["sprite_specs"] + secondary_frame["sprite_specs"]
    sprite_patches = [_make_sprite_patch(spec["mask"], spec["color"]) for spec in sprite_specs]

    frame_shapes = [tuple(map(int, frame["patch"].shape)) for frame in frames]
    sprite_shapes = [tuple(map(int, patch.shape)) for patch in sprite_patches]

    total_area = sum(h * w for h, w in frame_shapes + sprite_shapes)
    min_side = max(22, int(np.ceil(np.sqrt(total_area))) + 6)
    max_side = max(min_side, 30)

    for _ in range(80):
        h = random.randint(min_side, max_side)
        w = random.randint(min_side, max_side)
        if h < 8 or w < 8:
            continue

        frame_positions = _place_rectangles(frame_shapes, h, w, gap=1)
        if frame_positions is None:
            continue

        reserved = np.zeros((h, w), dtype=bool)
        for (r0, c0), (fh, fw) in zip(frame_positions, frame_shapes, strict=False):
            reserved[max(0, r0 - 1) : min(h, r0 + fh + 1), max(0, c0 - 1) : min(w, c0 + fw + 1)] = True

        sprite_positions: list[tuple[int, int]] = []
        ok = True
        order = sorted(range(len(sprite_patches)), key=lambda i: sprite_shapes[i][0] * sprite_shapes[i][1], reverse=True)
        temp_positions: list[tuple[int, int] | None] = [None] * len(sprite_patches)
        for idx in order:
            sh, sw = sprite_shapes[idx]
            choices: list[tuple[int, int]] = []
            for r0 in range(1, h - sh):
                for c0 in range(1, w - sw):
                    rr0 = max(0, r0 - 1)
                    rr1 = min(h, r0 + sh + 1)
                    cc0 = max(0, c0 - 1)
                    cc1 = min(w, c0 + sw + 1)
                    if not bool(reserved[rr0:rr1, cc0:cc1].any()):
                        choices.append((int(r0), int(c0)))
            if not choices:
                ok = False
                break
            r0, c0 = random.choice(choices)
            temp_positions[idx] = (int(r0), int(c0))
            reserved[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

        if not ok:
            continue
        sprite_positions = [(int(r), int(c)) for r, c in temp_positions]

        grid = np.zeros((h, w), dtype=int)
        frame_infos: list[dict] = []
        sprite_infos: list[dict] = []

        for frame, (r0, c0) in zip(frames, frame_positions, strict=False):
            patch = np.asarray(frame["patch"], dtype=int)
            _paste_nonzero(grid, patch, r0, c0)
            fh, fw = map(int, patch.shape)
            frame_infos.append(
                {
                    "bbox": (int(r0), int(c0), int(r0 + fh - 1), int(c0 + fw - 1)),
                    "ordered_colors": [int(c) for c in frame["ordered_colors"]],
                }
            )

        for patch, spec, (r0, c0) in zip(sprite_patches, sprite_specs, sprite_positions, strict=False):
            _paste_nonzero(grid, patch, r0, c0)
            sh, sw = map(int, patch.shape)
            sprite_infos.append(
                {
                    "bbox": (int(r0), int(c0), int(r0 + sh - 1), int(c0 + sw - 1)),
                    "mask": np.asarray(spec["mask"], dtype=bool),
                }
            )

        expected = _expected_output(grid, frame_infos, sprite_infos)
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
