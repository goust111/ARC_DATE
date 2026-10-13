from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
PALETTE = [2, 3, 4, 5, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_coords(coords)


def _label_mask(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    labels = np.zeros(mask.shape, dtype=int)
    dirs = DIR4 if int(connectivity) == 4 else DIR8
    next_id = 0
    h, w = mask.shape

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_id += 1
            stack = [(r, c)]
            labels[r, c] = next_id
            while stack:
                rr, cc = stack.pop()
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        stack.append((nr, nc))
    return labels, next_id


def _edge_ids(labels: np.ndarray, n_labels: int) -> set[int]:
    edge = set(map(int, labels[0, :]))
    edge.update(map(int, labels[-1, :]))
    edge.update(map(int, labels[:, 0]))
    edge.update(map(int, labels[:, -1]))
    edge.discard(0)
    return {k for k in edge if 1 <= k <= n_labels}


def _touches_mask8(mask: np.ndarray, r: int, c: int) -> bool:
    for dr, dc in DIR8:
        nr, nc = r + dr, c + dc
        if _in_bounds(mask.shape, nr, nc) and mask[nr, nc]:
            return True
    return False


def _fill_hole_from_anchor(
    hole_mask: np.ndarray,
    tile: np.ndarray,
    anchor_r: int,
    anchor_c: int,
) -> np.ndarray:
    h, w = hole_mask.shape
    tile_h, tile_w = tile.shape
    out = np.zeros((h, w), dtype=int)
    coords = np.argwhere(hole_mask)
    for r, c in coords:
        out[r, c] = int(tile[(int(r) - anchor_r) % tile_h, (int(c) - anchor_c) % tile_w])
    return out


def _enclosing_blue_object(
    component_mask: np.ndarray,
    blue_labels: np.ndarray,
) -> int | None:
    ids: set[int] = set()
    coords = np.argwhere(component_mask)
    for r, c in coords:
        for dr, dc in DIR4:
            nr, nc = int(r) + dr, int(c) + dc
            if not _in_bounds(component_mask.shape, nr, nc) or component_mask[nr, nc]:
                continue
            obj = int(blue_labels[nr, nc])
            if obj == 0:
                return None
            ids.add(obj)
    if len(ids) != 1:
        return None
    return next(iter(ids))


def _rank_unique(counts: Counter[int]) -> tuple[int, int]:
    ranked = counts.most_common()
    assert len(ranked) >= 2, "expected at least two non-black colors"
    assert ranked[0][1] > ranked[1][1], "expected a unique most frequent color"
    assert ranked[-2][1] > ranked[-1][1], "expected a unique least frequent color"
    return int(ranked[0][0]), int(ranked[-1][0])


def _analyze_scene(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    blue_mask = g == BLUE
    blue_labels, _n_blue = _label_mask(blue_mask, connectivity=4)
    non_blue_labels, n_non_blue = _label_mask(g != BLUE, connectivity=4)
    black_labels, n_black = _label_mask(g == BLACK, connectivity=4)
    external_black_ids = _edge_ids(black_labels, n_black)
    external_black = np.isin(black_labels, list(external_black_ids))

    infos: list[dict] = []
    for comp_id in range(1, n_non_blue + 1):
        component_mask = non_blue_labels == comp_id
        coords = np.argwhere(component_mask)
        if coords.size == 0:
            continue
        if np.any(
            (coords[:, 0] == 0)
            | (coords[:, 1] == 0)
            | (coords[:, 0] == g.shape[0] - 1)
            | (coords[:, 1] == g.shape[1] - 1)
        ):
            continue

        object_id = _enclosing_blue_object(component_mask, blue_labels)
        if object_id is None:
            continue

        seed_mask = component_mask & (g != BLACK)
        if not seed_mask.any():
            continue

        hole_bbox = _bbox(component_mask)
        seed_bbox = _bbox(seed_mask)
        assert hole_bbox is not None and seed_bbox is not None
        hr0, hc0, hr1, hc1 = hole_bbox
        sr0, sc0, sr1, sc1 = seed_bbox

        tile = g[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
        seed_colors = [c for c in sorted(set(map(int, tile.ravel()))) if c not in (BLACK, BLUE)]
        assert len(seed_colors) == 2, "each seed must use exactly two non-black non-blue colors"

        hole_h = hr1 - hr0 + 1
        hole_w = hc1 - hc0 + 1
        tile_h, tile_w = tile.shape
        assert hole_h % tile_h == 0 and hole_w % tile_w == 0, "hole bbox must be a multiple of the seed bbox"

        local_hole = component_mask[hr0 : hr1 + 1, hc0 : hc1 + 1]
        filled_local = _fill_hole_from_anchor(local_hole, tile, 0, 0)
        filled_counts = Counter(
            int(filled_local[r - hr0, c - hc0])
            for r, c in coords
            if int(filled_local[r - hr0, c - hc0]) != BLACK
        )
        high, low = _rank_unique(filled_counts)

        seed_counts = Counter(int(v) for v in tile.ravel() if int(v) not in (BLACK, BLUE))
        seed_high, seed_low = _rank_unique(seed_counts)

        obj_mask = blue_labels == object_id
        inner_boundary = np.array(
            [(int(r), int(c)) for r, c in coords if _touches_mask8(obj_mask, int(r), int(c))],
            dtype=int,
        )
        outer_boundary = np.array(
            [
                (r, c)
                for r, c in np.argwhere(external_black)
                if _touches_mask8(obj_mask, int(r), int(c))
            ],
            dtype=int,
        )
        core_area = int(component_mask.sum()) - len(inner_boundary)

        infos.append(
            {
                "object_id": object_id,
                "object_mask": obj_mask,
                "component_mask": component_mask,
                "coords": coords,
                "hole_bbox": hole_bbox,
                "seed_bbox": seed_bbox,
                "tile": tile,
                "high": high,
                "low": low,
                "seed_high": seed_high,
                "seed_low": seed_low,
                "inner_boundary": inner_boundary,
                "outer_boundary": outer_boundary,
                "core_area": core_area,
            }
        )
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """Tile each enclosed blue cavity, then color the ring's outer and inner sides from the tiled color frequencies.

    Input structure:
    - The background is black and the active wall color is blue.
    - There are 2-3 blue 4-connected closed ring objects with irregular one-cell walls, each enclosing one non-blue seed inside a cavity.
    - The seed inside each cavity is a 2x2 to 3x3 two-color pattern, and some separate blue open objects may also appear.

    Rule:
    - Treat each non-edge 4-connected non-blue region whose whole 4-neighbor boundary belongs to one blue object as one cavity.
    - Take the bbox of the cavity's non-black seed, align that seed bbox to the cavity bbox top-left, and tile it across the whole cavity while keeping seed zeros black.
    - Count the non-black colors in the tiled cavity; color the external black pixels that are 8-neighbor to the enclosing blue ring with the unique most frequent color, and recolor the cavity pixels that are 8-neighbor to that ring with the unique least frequent color.
    - Keep the blue wall itself unchanged and leave open blue objects untouched.

    Key insight:
    - The border colors are not fixed: they are computed from the full tiled cavity, so the cavity shape and tiling phase relative to the cavity bbox both matter before the ring can be recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    infos = _analyze_scene(g)
    out = g.copy()

    for info in infos:
        r0, c0, r1, c1 = info["hole_bbox"]
        tile = info["tile"]
        tile_h, tile_w = tile.shape
        for r, c in info["coords"]:
            out[r, c] = int(tile[(int(r) - r0) % tile_h, (int(c) - c0) % tile_w])

    for info in infos:
        high = int(info["high"])
        low = int(info["low"])
        if len(info["outer_boundary"]) > 0:
            out[info["outer_boundary"][:, 0], info["outer_boundary"][:, 1]] = high
        if len(info["inner_boundary"]) > 0:
            out[info["inner_boundary"][:, 0], info["inner_boundary"][:, 1]] = low

    return out


GEN_ATTEMPTS = 600
GRID_SIDE_RANGE = (22, 28)
RING_COUNT_CHOICES = (2, 3)
SEED_SHAPES = [(2, 2), (2, 3), (3, 2), (3, 3)]
REPEAT_CHOICES = (2, 3)


def _connected(mask: np.ndarray) -> bool:
    _, n = _label_mask(mask, connectivity=4)
    return n == 1


def _make_irregular_hole(h: int, w: int) -> np.ndarray | None:
    for _ in range(80):
        mask = np.ones((h, w), dtype=bool)
        carved = False
        for _ in range(random.randint(1, 3)):
            side = random.choice(["top", "bottom", "left", "right"])
            trial = mask.copy()
            if side in {"top", "bottom"}:
                depth_max = min(2, h - 2)
                if depth_max < 1:
                    continue
                depth = random.randint(1, depth_max)
                notch_w = random.randint(1, w - 1)
                c0 = random.randint(0, w - notch_w)
                rows = slice(0, depth) if side == "top" else slice(h - depth, h)
                trial[rows, c0 : c0 + notch_w] = False
            else:
                depth_max = min(2, w - 2)
                if depth_max < 1:
                    continue
                depth = random.randint(1, depth_max)
                notch_h = random.randint(1, h - 1)
                r0 = random.randint(0, h - notch_h)
                cols = slice(0, depth) if side == "left" else slice(w - depth, w)
                trial[r0 : r0 + notch_h, cols] = False
            if trial.any() and _connected(trial):
                mask = trial
                carved = carved or (mask.sum() < h * w)

        if not carved:
            continue
        bb = _bbox(mask)
        if bb != (0, 0, h - 1, w - 1):
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        return mask
    return None


def _build_ring_patch(hole_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    h, w = hole_mask.shape
    patch = np.zeros((h + 4, w + 4), dtype=int)
    local_hole = np.zeros_like(patch, dtype=bool)
    local_hole[2 : 2 + h, 2 : 2 + w] = hole_mask

    ring = np.zeros_like(patch, dtype=bool)
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            if local_hole[r, c]:
                continue
            # Use the 8-neighbor shell around the cavity so diagonal corner caps
            # keep the blue wall itself 4-connected.
            for dr, dc in DIR8:
                nr, nc = r + dr, c + dc
                if _in_bounds(patch.shape, nr, nc) and local_hole[nr, nc]:
                    ring[r, c] = True
                    break
    if not _connected(ring):
        return None
    patch[ring] = BLUE
    return patch, ring, local_hole


def _seed_positions(hole_mask: np.ndarray, sh: int, sw: int) -> list[tuple[int, int]]:
    h, w = hole_mask.shape
    pos = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if hole_mask[r0 : r0 + sh, c0 : c0 + sw].all():
                pos.append((r0, c0))
    return pos


def _random_tile(sh: int, sw: int, color_a: int, color_b: int) -> np.ndarray:
    area = sh * sw
    count_a = random.randint(1, area - 1)
    if 2 * count_a == area:
        count_a = 1 if count_a != 1 else area - 1
    cells = [(r, c) for r in range(sh) for c in range(sw)]
    random.shuffle(cells)
    tile = np.full((sh, sw), color_b, dtype=int)
    for r, c in cells[:count_a]:
        tile[r, c] = color_a
    return tile


def _tile_stats(hole_mask: np.ndarray, tile: np.ndarray, seed_pos: tuple[int, int]) -> dict:
    rect_fill = _fill_hole_from_anchor(hole_mask, tile, 0, 0)
    seed_fill = _fill_hole_from_anchor(hole_mask, tile, seed_pos[0], seed_pos[1])
    actual_counts = Counter(int(rect_fill[r, c]) for r, c in np.argwhere(hole_mask))
    seed_counts = Counter(int(v) for v in tile.ravel())
    actual_counts.pop(BLACK, None)
    seed_counts.pop(BLACK, None)
    actual_high, actual_low = _rank_unique(actual_counts)
    seed_high, seed_low = _rank_unique(seed_counts)
    return {
        "actual_high": actual_high,
        "actual_low": actual_low,
        "seed_high": seed_high,
        "seed_low": seed_low,
        "anchor_diff": bool(np.any(rect_fill[hole_mask] != seed_fill[hole_mask])),
    }


def _transform_patch(patch: np.ndarray) -> np.ndarray:
    out = np.asarray(patch, dtype=int)
    if random.random() < 0.8:
        out = np.rot90(out, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.asarray(out, dtype=int)


def _special_flip_ring_patch(color_a: int, color_b: int) -> tuple[np.ndarray, dict] | None:
    hole_mask = np.array(
        [
            [1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1],
        ],
        dtype=bool,
    )
    built = _build_ring_patch(hole_mask)
    if built is None:
        return None
    patch, _ring_mask, _local_hole = built
    seed_pos = (1, 1)
    tile = np.array(
        [
            [color_a, color_a, color_a],
            [color_a, color_b, color_b],
        ],
        dtype=int,
    )
    stats = _tile_stats(hole_mask, tile, seed_pos)
    assert stats["actual_high"] == color_b and stats["seed_high"] == color_a and stats["anchor_diff"]

    patch_in = patch.copy()
    sh, sw = tile.shape
    patch_in[2 + seed_pos[0] : 2 + seed_pos[0] + sh, 2 + seed_pos[1] : 2 + seed_pos[1] + sw] = tile
    patch_in = _transform_patch(patch_in)
    return patch_in, {
        "high": stats["actual_high"],
        "low": stats["actual_low"],
        "seed_high": stats["seed_high"],
        "seed_low": stats["seed_low"],
        "seed_pos": seed_pos,
        "anchor_diff": stats["anchor_diff"],
        "flip": True,
    }


def _ring_input_patch(
    color_a: int,
    color_b: int,
    require_flip: bool,
    require_anchor_diff: bool,
) -> tuple[np.ndarray, dict] | None:
    shape_pool = SEED_SHAPES[:]
    random.shuffle(shape_pool)
    if require_flip:
        shape_pool.sort(key=lambda x: -(x[0] * x[1]))

    for sh, sw in shape_pool:
        for _ in range(120):
            h = sh * random.choice(REPEAT_CHOICES)
            w = sw * random.choice(REPEAT_CHOICES)
            hole_mask = _make_irregular_hole(h, w)
            if hole_mask is None:
                continue
            built = _build_ring_patch(hole_mask)
            if built is None:
                continue
            patch, _ring_mask, local_hole = built
            positions = _seed_positions(hole_mask, sh, sw)
            if require_anchor_diff:
                positions = [p for p in positions if p != (0, 0) and ((p[0] % sh) != 0 or (p[1] % sw) != 0)]
            if not positions:
                continue

            for _ in range(160):
                seed_pos = random.choice(positions)
                tile = _random_tile(sh, sw, color_a, color_b)
                try:
                    stats = _tile_stats(hole_mask, tile, seed_pos)
                except AssertionError:
                    continue
                if require_flip and stats["actual_high"] == stats["seed_high"]:
                    continue
                if require_anchor_diff and not stats["anchor_diff"]:
                    continue

                patch_in = patch.copy()
                sr, sc = seed_pos
                patch_in[2 + sr : 2 + sr + sh, 2 + sc : 2 + sc + sw] = tile
                component_mask = local_hole.copy()
                inner_boundary = np.array(
                    [
                        (r, c)
                        for r, c in np.argwhere(component_mask)
                        if _touches_mask8(patch == BLUE, int(r), int(c))
                    ],
                    dtype=int,
                )
                core_area = int(component_mask.sum()) - len(inner_boundary)
                if core_area <= 0:
                    continue

                info = {
                    "input_patch": patch_in,
                    "high": stats["actual_high"],
                    "low": stats["actual_low"],
                    "seed_high": stats["seed_high"],
                    "seed_low": stats["seed_low"],
                    "seed_pos": seed_pos,
                    "anchor_diff": stats["anchor_diff"],
                    "flip": stats["actual_high"] != stats["seed_high"],
                }
                return patch_in, info
    return None


def _open_blue_patch() -> np.ndarray:
    choices = [
        np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int),
        np.array([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 1, 0]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=int),
        np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=int),
    ]
    patch = random.choice(choices).copy()
    if random.random() < 0.5:
        patch = np.rot90(patch, random.randint(0, 3))
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    return patch.astype(int)


def _find_positions(
    grid_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    gap: int,
) -> list[tuple[int, int]]:
    h, w = grid_shape
    ph, pw = patch_shape
    pos = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            r1, c1 = r0 + ph - 1, c0 + pw - 1
            ok = True
            for br0, bc0, br1, bc1 in boxes:
                if not (r1 + gap < br0 or br1 + gap < r0 or c1 + gap < bc0 or bc1 + gap < c0):
                    ok = False
                    break
            if ok:
                pos.append((r0, c0))
    return pos


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BLACK
    view = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    view[mask] = patch[mask]


def _scene_ok(grid: np.ndarray) -> bool:
    infos = _analyze_scene(grid)
    if len(infos) not in RING_COUNT_CHOICES:
        return False
    if not all(int(info["component_mask"].sum()) < (info["hole_bbox"][2] - info["hole_bbox"][0] + 1) * (info["hole_bbox"][3] - info["hole_bbox"][1] + 1) for info in infos):
        return False
    if not all(info["core_area"] > 0 for info in infos):
        return False
    if len({int(info["high"]) for info in infos}) != len(infos):
        return False
    if not any(int(info["high"]) != int(info["seed_high"]) for info in infos):
        return False
    if not any(
        ((info["seed_bbox"][0] - info["hole_bbox"][0]) != 0 or (info["seed_bbox"][1] - info["hole_bbox"][1]) != 0)
        and np.any(
            _fill_hole_from_anchor(
                info["component_mask"][info["hole_bbox"][0] : info["hole_bbox"][2] + 1, info["hole_bbox"][1] : info["hole_bbox"][3] + 1],
                info["tile"],
                0,
                0,
            )[info["component_mask"][info["hole_bbox"][0] : info["hole_bbox"][2] + 1, info["hole_bbox"][1] : info["hole_bbox"][3] + 1]]
            != _fill_hole_from_anchor(
                info["component_mask"][info["hole_bbox"][0] : info["hole_bbox"][2] + 1, info["hole_bbox"][1] : info["hole_bbox"][3] + 1],
                info["tile"],
                info["seed_bbox"][0] - info["hole_bbox"][0],
                info["seed_bbox"][1] - info["hole_bbox"][1],
            )[info["component_mask"][info["hole_bbox"][0] : info["hole_bbox"][2] + 1, info["hole_bbox"][1] : info["hole_bbox"][3] + 1]]
        )
        for info in infos
    ):
        return False
    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        ring_count = random.choice(RING_COUNT_CHOICES)
        rows = random.randint(*GRID_SIDE_RANGE)
        cols = random.randint(*GRID_SIDE_RANGE)
        grid = np.zeros((rows, cols), dtype=int)

        colors = random.sample(PALETTE, 2 * ring_count)
        ring_specs = []
        for idx in range(ring_count):
            if idx == 0:
                built = _special_flip_ring_patch(colors[2 * idx], colors[2 * idx + 1])
            else:
                built = _ring_input_patch(
                    colors[2 * idx],
                    colors[2 * idx + 1],
                    require_flip=False,
                    require_anchor_diff=False,
                )
            if built is None:
                ring_specs = []
                break
            ring_specs.append(built)
        if not ring_specs:
            continue

        boxes: list[tuple[int, int, int, int]] = []
        patches = [spec[0] for spec in ring_specs]
        random.shuffle(patches)
        ok = True
        for patch in sorted(patches, key=lambda x: x.shape[0] * x.shape[1], reverse=True):
            pos = _find_positions(grid.shape, patch.shape, boxes, gap=1)
            if not pos:
                ok = False
                break
            r0, c0 = random.choice(pos)
            _place_patch(grid, patch, r0, c0)
            boxes.append((r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1))
        if not ok:
            continue

        if random.random() < 0.7:
            distractor = _open_blue_patch()
            pos = _find_positions(grid.shape, distractor.shape, boxes, gap=2)
            if pos:
                r0, c0 = random.choice(pos)
                _place_patch(grid, distractor, r0, c0)

        if _scene_ok(grid):
            return grid

    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
