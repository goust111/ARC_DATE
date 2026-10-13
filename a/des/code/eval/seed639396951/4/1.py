from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    border = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
    values, counts = np.unique(border, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for candidate in (rotated, np.fliplr(rotated)):
            tight = _trim_mask(candidate)
            key = (int(tight.shape[0]), int(tight.shape[1]), tight.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(tight)
    return variants


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min(
        (int(variant.shape[0]), int(variant.shape[1]), variant.tobytes())
        for variant in _mask_variants(mask)
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not bool(seen[nr, nc])
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            _, bbox = _coords_to_mask(arr)
            comps.append({"color": color, "coords": arr, "bbox": bbox})
    return comps


def _interior_holes(subgrid: np.ndarray, bg: int) -> list[np.ndarray]:
    holes: list[np.ndarray] = []
    for comp in _components(subgrid == bg):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 1] == 0):
            continue
        if np.any(comp[:, 0] == subgrid.shape[0] - 1) or np.any(comp[:, 1] == subgrid.shape[1] - 1):
            continue
        holes.append(comp)
    return holes


def _frame_info(grid: np.ndarray, comp: dict, bg: int) -> dict | None:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]

    if not np.all((sub == bg) | (sub == color)):
        return None
    if not np.all(sub[0, :] == color) or not np.all(sub[-1, :] == color):
        return None
    if not np.all(sub[:, 0] == color) or not np.all(sub[:, -1] == color):
        return None

    holes = _interior_holes(sub, bg)
    if len(holes) != 1:
        return None

    hole_mask, hole_bbox = _coords_to_mask(holes[0])
    hr0, hc0, _hr1, _hc1 = hole_bbox
    return {
        "color": color,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "hole_coords": holes[0] + np.array([[r0, c0]], dtype=int),
        "hole_offset": (int(hr0), int(hc0)),
        "hole_mask": hole_mask,
        "key": _shape_key(hole_mask),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each enclosed black window inside a large colored rectangle with an outside solid sprite
    of the same shape up to rotation/flip, then fill the window with that sprite's color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    frames: list[dict] = []
    sprites: list[dict] = []
    for comp in _color_components(g, bg):
        frame = _frame_info(g, comp, bg)
        if frame is not None:
            frames.append(frame)
            continue
        sprite_mask, _ = _coords_to_mask(comp["coords"])
        sprites.append(
            {
                "color": int(comp["color"]),
                "coords": comp["coords"],
                "mask": sprite_mask,
                "key": _shape_key(sprite_mask),
            }
        )

    sprite_map: dict[tuple[int, int, bytes], dict] = {}
    for sprite in sprites:
        key = sprite["key"]
        assert key not in sprite_map, "sprite shapes must be unique"
        sprite_map[key] = sprite

    out = g.copy()
    for sprite in sprites:
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = bg

    for frame in frames:
        match = sprite_map.get(frame["key"])
        assert match is not None, "every frame hole must have a matching sprite"
        out[frame["hole_coords"][:, 0], frame["hole_coords"][:, 1]] = int(match["color"])

    return out


def _shape(mask_rows: Iterable[str]) -> np.ndarray:
    rows = [[ch == "1" for ch in row] for row in mask_rows]
    return np.array(rows, dtype=bool)


_BASE_SHAPES = [
    _shape(["10", "11"]),
    _shape(["111", "010"]),
    _shape(["011", "110"]),
    _shape(["10", "10", "11"]),
    _shape(["11", "11", "10"]),
    _shape(["010", "111"]),
    _shape(["101", "111"]),
    _shape(["011", "110", "010"]),
    _shape(["100", "110", "011"]),
    _shape(["110", "011", "001"]),
    _shape(["010", "111", "010"]),
    _shape(["11", "01", "01"]),
]


def _unique_library() -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for mask in _BASE_SHAPES:
        key = _shape_key(mask)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(mask)
    return shapes


_SHAPE_LIBRARY = _unique_library()


def _random_variant(mask: np.ndarray) -> np.ndarray:
    return random.choice(_mask_variants(mask)).copy()


def _make_frame_patch(hole_mask: np.ndarray, frame_color: int, margins: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    top, bottom, left, right = margins
    hole_h, hole_w = hole_mask.shape
    patch = np.full((top + hole_h + bottom, left + hole_w + right), frame_color, dtype=int)
    patch[top : top + hole_h, left : left + hole_w][hole_mask] = BG
    hole_coords = np.argwhere(hole_mask) + np.array([[top, left]], dtype=int)
    return patch, hole_coords


def _render_shape(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, color, BG).astype(int)


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    H, W = occupied.shape
    if r0 < 1 or c0 < 1 or r0 + h > H - 1 or c0 + w > W - 1:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _place_patch(
    canvas: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    pad: int,
) -> tuple[int, int] | None:
    H, W = canvas.shape
    h, w = patch.shape
    candidates = [(r, c) for r in range(1, H - h) for c in range(1, W - w)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        if not _can_place(occupied, int(r0), int(c0), h, w, pad):
            continue
        region = canvas[r0 : r0 + h, c0 : c0 + w]
        region[patch != BG] = patch[patch != BG]
        occupied[r0 : r0 + h, c0 : c0 + w] = True
        return int(r0), int(c0)
    return None


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_frames = random.randint(2, 4)
    bases = random.sample(_SHAPE_LIBRARY, n_frames)
    colors = random.sample(list(range(1, 10)), 2 * n_frames)
    frame_colors = colors[:n_frames]
    sprite_colors = colors[n_frames:]

    descriptors: list[dict] = []
    for base, frame_color, sprite_color in zip(bases, frame_colors, sprite_colors, strict=False):
        hole_mask = _random_variant(base)
        sprite_mask = _random_variant(base)
        margins = (
            random.randint(1, 3),
            random.randint(1, 3),
            random.randint(1, 3),
            random.randint(1, 3),
        )
        frame_patch, hole_coords = _make_frame_patch(hole_mask, int(frame_color), margins)
        sprite_patch = _render_shape(sprite_mask, int(sprite_color))
        descriptors.append(
            {
                "frame_patch": frame_patch,
                "hole_coords": hole_coords,
                "sprite_patch": sprite_patch,
                "sprite_color": int(sprite_color),
            }
        )

    max_dim = max(max(d["frame_patch"].shape) for d in descriptors)
    total_bbox_area = sum(
        (int(d["frame_patch"].shape[0]) + 2) * (int(d["frame_patch"].shape[1]) + 2)
        + (int(d["sprite_patch"].shape[0]) + 2) * (int(d["sprite_patch"].shape[1]) + 2)
        for d in descriptors
    )
    min_side = max(18, max_dim + 4, int(np.ceil(np.sqrt(total_bbox_area))) + 2)
    max_side = max(min_side, min(30, min_side + 8))
    H = random.randint(min_side, max_side)
    W = random.randint(min_side, max_side)

    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    for desc in sorted(descriptors, key=lambda d: -int(np.prod(d["frame_patch"].shape))):
        pos = _place_patch(out, occupied, desc["frame_patch"], pad=1)
        if pos is None:
            return None
        r0, c0 = pos
        desc["frame_origin"] = (int(r0), int(c0))
        region = inp[r0 : r0 + desc["frame_patch"].shape[0], c0 : c0 + desc["frame_patch"].shape[1]]
        region[desc["frame_patch"] != BG] = desc["frame_patch"][desc["frame_patch"] != BG]

    for desc in descriptors:
        r0, c0 = desc["frame_origin"]
        hole_coords = desc["hole_coords"] + np.array([[r0, c0]], dtype=int)
        out[hole_coords[:, 0], hole_coords[:, 1]] = int(desc["sprite_color"])

    for desc in sorted(descriptors, key=lambda d: -int(np.prod(d["sprite_patch"].shape))):
        pos = _place_patch(inp, occupied, desc["sprite_patch"], pad=1)
        if pos is None:
            return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sample = _sample_case()
        if sample is None:
            continue
        inp, expected = sample
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid ARC sample")
