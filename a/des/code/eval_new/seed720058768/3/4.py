from __future__ import annotations

import random

import numpy as np


BG = 0
BODY = 5
MARKER = 9
PALETTE_POOL = (1, 2, 3, 4, 6, 7, 8)
BASE_MASK = np.array(
    [
        [0, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 0, 0, 1],
    ],
    dtype=bool,
)

OBJECT_COUNT_RANGE = (2, 4)
SCALE_CHOICES = (1, 2, 3, 4)
OUTPUT_H_RANGE = (14, 29)
OUTPUT_W_RANGE = (14, 30)
GEN_RETRIES = 400
PLACE_RETRIES = 250


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _symmetry_axis(mask: np.ndarray) -> str:
    vertical = bool(np.array_equal(mask, np.fliplr(mask)))
    horizontal = bool(np.array_equal(mask, np.flipud(mask)))
    assert vertical ^ horizontal, "each object must have exactly one symmetry axis"
    return "vertical" if vertical else "horizontal"


def _mirror_local(pos: tuple[int, int], shape: tuple[int, int], axis: str) -> tuple[int, int]:
    r, c = pos
    h, w = shape
    if axis == "vertical":
        return int(r), int(w - 1 - c)
    return int(h - 1 - r), int(c)


def _find_palette_row(grid: np.ndarray, bg: int) -> tuple[int, list[int]]:
    H, _W = grid.shape
    candidates: list[tuple[int, int, list[int]]] = []
    for r in (0, H - 1):
        cols = np.flatnonzero(grid[r] != bg)
        if cols.size < 2:
            continue
        c0 = int(cols[0])
        c1 = int(cols[-1])
        if c1 - c0 + 1 != int(cols.size):
            continue
        seg = [int(v) for v in grid[r, c0 : c1 + 1]]
        if bg in seg or len(set(seg)) != len(seg):
            continue
        if any(int(np.count_nonzero(grid == color)) != 1 for color in seg):
            continue
        adj = 1 if r == 0 else H - 2
        if np.any(grid[adj, c0 : c1 + 1] != bg):
            continue
        candidates.append((len(seg), int(r), seg))
    assert len(candidates) == 1, "expected one edge palette row"
    _length, row, palette = candidates[0]
    return int(row), palette


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor similarly shaped objects by area rank and mirror each internal marker across that object's own symmetry axis.

    Input structure:
    - bg = background. Several connected objects share one body color and differ only by scale and axis orientation.
    - Each object contains exactly one off-axis marker pixel of a second color.
    - Each object has exactly one symmetry axis, either horizontal or vertical.
    - A 1xK color bar sits on a grid edge and lists the replacement colors from smallest object to largest.

    Rule:
    - Remove the edge color bar from the scene.
    - Rank the objects by body area from small to large and recolor each whole body with the matching color-bar entry.
    - For each object, detect whether its body is horizontally or vertically symmetric and reflect its marker to the mirrored in-object position.
    - The original marker cell becomes body-colored; the mirrored cell stays marker-colored.

    Key insight:
    - The color bar is a size-order legend, while the marker motion is local to each object and depends on that object's own symmetry axis rather than a single global direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    palette_row, palette = _find_palette_row(g, bg)
    scene = np.delete(g, palette_row, axis=0)

    comps = _connected_components(scene != bg)
    assert len(comps) == len(palette) >= 2

    objects = []
    for coords in comps:
        colors, counts = np.unique(scene[coords[:, 0], coords[:, 1]], return_counts=True)
        assert len(colors) == 2
        order = np.argsort(counts)
        marker_color = int(colors[order[0]])
        marker_count = int(counts[order[0]])
        body_count = int(counts[order[1]])
        assert marker_count == 1 and body_count > 1

        marker_r, marker_c = map(int, coords[scene[coords[:, 0], coords[:, 1]] == marker_color][0])
        mask, (r0, c0, _r1, _c1) = _mask_from_coords(coords)
        axis = _symmetry_axis(mask)
        local_marker = (int(marker_r - r0), int(marker_c - c0))
        mirrored_local = _mirror_local(local_marker, mask.shape, axis)
        assert mask[mirrored_local]

        objects.append(
            {
                "coords": coords,
                "body_area": body_count,
                "marker_color": marker_color,
                "mirrored_global": (int(r0 + mirrored_local[0]), int(c0 + mirrored_local[1])),
            }
        )

    areas = [obj["body_area"] for obj in objects]
    assert len(set(areas)) == len(areas)
    objects.sort(key=lambda obj: int(obj["body_area"]))

    out = np.full(scene.shape, bg, dtype=int)
    for rank, obj in enumerate(objects):
        color = int(palette[rank])
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = color
        mr, mc = obj["mirrored_global"]
        out[int(mr), int(mc)] = int(obj["marker_color"])
    return out


def _scaled_mask(scale: int, axis: str) -> np.ndarray:
    mask = np.kron(BASE_MASK, np.ones((scale, scale), dtype=bool))
    if axis == "horizontal":
        mask = np.rot90(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
    else:
        if random.random() < 0.5:
            mask = np.flipud(mask)
    return np.asarray(mask, dtype=bool)


def _interior_candidates(mask: np.ndarray, axis: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    interior = np.zeros_like(mask, dtype=bool)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if mask[r, c] and mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]:
                interior[r, c] = True

    candidates: list[tuple[int, int]] = []
    for r, c in np.argwhere(interior):
        rr, cc = _mirror_local((int(r), int(c)), mask.shape, axis)
        if (rr, cc) != (int(r), int(c)) and interior[rr, cc]:
            candidates.append((int(r), int(c)))
    return candidates


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _paste_mask(canvas: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    h, w = mask.shape
    canvas[r0 : r0 + h, c0 : c0 + w][mask] = int(color)


def _make_object(scale: int, axis: str) -> dict:
    mask = _scaled_mask(scale, axis)
    assert _symmetry_axis(mask) == axis
    marker_candidates = _interior_candidates(mask, axis)
    assert marker_candidates
    marker_local = random.choice(marker_candidates)
    mirrored_local = _mirror_local(marker_local, mask.shape, axis)
    return {
        "scale": int(scale),
        "axis": axis,
        "mask": mask,
        "marker_local": marker_local,
        "mirrored_local": mirrored_local,
        "body_area": int(mask.sum() - 1),
    }


def _place_objects(objects: list[dict], out_h: int, out_w: int) -> list[dict] | None:
    occupied = np.zeros((out_h, out_w), dtype=bool)
    placed: list[dict] = []
    for obj in sorted(objects, key=lambda item: int(item["mask"].sum()), reverse=True):
        mask = obj["mask"]
        h, w = mask.shape
        if h > out_h or w > out_w:
            return None
        placed_here = False
        for _ in range(PLACE_RETRIES):
            r0 = random.randint(0, out_h - h)
            c0 = random.randint(0, out_w - w)
            if (_dilate(occupied)[r0 : r0 + h, c0 : c0 + w] & mask).any():
                continue
            occupied[r0 : r0 + h, c0 : c0 + w] |= mask
            entry = dict(obj)
            entry["r0"] = int(r0)
            entry["c0"] = int(c0)
            placed.append(entry)
            placed_here = True
            break
        if not placed_here:
            return None
    return placed


def _render_input(placed: list[dict], out_h: int, out_w: int, palette: list[int], palette_start: int) -> np.ndarray:
    scene = np.full((out_h, out_w), BG, dtype=int)
    for obj in placed:
        _paste_mask(scene, obj["mask"], BODY, int(obj["r0"]), int(obj["c0"]))
        mr = int(obj["r0"] + obj["marker_local"][0])
        mc = int(obj["c0"] + obj["marker_local"][1])
        scene[mr, mc] = MARKER

    inp = np.full((out_h + 1, out_w), BG, dtype=int)
    inp[:-1] = scene
    inp[-1, palette_start : palette_start + len(palette)] = np.asarray(palette, dtype=int)
    return inp


def _expected_output(placed: list[dict], out_h: int, out_w: int, palette: list[int]) -> np.ndarray:
    out = np.full((out_h, out_w), BG, dtype=int)
    ranked = sorted(placed, key=lambda obj: int(obj["body_area"]))
    for rank, obj in enumerate(ranked):
        color = int(palette[rank])
        _paste_mask(out, obj["mask"], color, int(obj["r0"]), int(obj["c0"]))
        mr = int(obj["r0"] + obj["mirrored_local"][0])
        mc = int(obj["c0"] + obj["mirrored_local"][1])
        out[mr, mc] = MARKER
    return out


def _gen_attempt() -> np.ndarray | None:
    k = random.randint(*OBJECT_COUNT_RANGE)
    scales = sorted(random.sample(SCALE_CHOICES, k))
    palette = random.sample(PALETTE_POOL, k)

    axes = [random.choice(("vertical", "horizontal")) for _ in range(k)]
    if len(set(axes)) == 1:
        axes[random.randrange(k)] = "horizontal" if axes[0] == "vertical" else "vertical"

    objects = [_make_object(scale, axis) for scale, axis in zip(scales, axes)]
    max_h = max(int(obj["mask"].shape[0]) for obj in objects)
    max_w = max(int(obj["mask"].shape[1]) for obj in objects)

    h_lo = max(OUTPUT_H_RANGE[0], max_h + 1)
    w_lo = max(OUTPUT_W_RANGE[0], max(max_w, k) + 1)
    if h_lo > OUTPUT_H_RANGE[1] or w_lo > OUTPUT_W_RANGE[1]:
        return None

    out_h = random.randint(h_lo, OUTPUT_H_RANGE[1])
    out_w = random.randint(w_lo, OUTPUT_W_RANGE[1])
    placed = _place_objects(objects, out_h, out_w)
    if placed is None:
        return None

    scene = np.full((out_h, out_w), BG, dtype=int)
    for obj in placed:
        _paste_mask(scene, obj["mask"], BODY, int(obj["r0"]), int(obj["c0"]))
        mr = int(obj["r0"] + obj["marker_local"][0])
        mc = int(obj["c0"] + obj["marker_local"][1])
        scene[mr, mc] = MARKER

    starts = [c for c in range(out_w - k + 1) if np.all(scene[-1, c : c + k] == BG)]
    if not starts:
        return None
    palette_start = random.choice(starts)

    inp = _render_input(placed, out_h, out_w, palette, palette_start)
    expected = _expected_output(placed, out_h, out_w, palette)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
