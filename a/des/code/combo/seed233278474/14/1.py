import random
from collections import Counter

import numpy as np


BG = 0
PIECE_COLOR = 5
FRAME_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator configuration.
GRID_H_RANGE = (24, 30)
GRID_W_RANGE = (24, 30)
FRAME_COUNT_OPTIONS = [1, 2, 2, 3]
MAX_HOLES_PER_FRAME = 3
TOTAL_HOLES_MAX = 5
POLYOMINO_AREA_RANGE = (3, 7)
POLYOMINO_BOX_MAX = 5
FRAME_PACK_TRIES = 120
GLOBAL_FRAME_PLACE_TRIES = 160
GLOBAL_PIECE_PLACE_TRIES = 160
GENERATE_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r_idx = np.flatnonzero(rows)
    c_idx = np.flatnonzero(cols)
    return mask[r_idx[0] : r_idx[-1] + 1, c_idx[0] : c_idx[-1] + 1]


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _rotation_variants(mask: np.ndarray) -> list[np.ndarray]:
    base = _trim(np.asarray(mask, dtype=bool))
    seen = set()
    variants = []
    for k in range(4):
        rot = _trim(np.rot90(base, k))
        key = (rot.shape, np.ascontiguousarray(rot.astype(np.uint8)).tobytes())
        if key in seen:
            continue
        seen.add(key)
        variants.append(rot)
    return variants


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    best = None
    for rot in _rotation_variants(mask):
        key = (rot.shape, np.ascontiguousarray(rot.astype(np.uint8)).tobytes())
        if best is None or key < best:
            best = key
    return best


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    frame_mask = (grid != BG) & (grid != PIECE_COLOR)
    for coords in _components(frame_mask):
        colors = np.unique(grid[coords[:, 0], coords[:, 1]])
        if len(colors) != 1:
            continue
        color = int(colors[0])
        r0, c0, r1, c1 = _bbox(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        holes = []
        for hole_coords in _components(sub == BG):
            if (
                np.any(hole_coords[:, 0] == 0)
                or np.any(hole_coords[:, 0] == sub.shape[0] - 1)
                or np.any(hole_coords[:, 1] == 0)
                or np.any(hole_coords[:, 1] == sub.shape[1] - 1)
            ):
                continue
            hr0, hc0, hr1, hc1 = _bbox(hole_coords)
            hole_mask = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
            hole_mask[hole_coords[:, 0] - hr0, hole_coords[:, 1] - hc0] = True
            holes.append(
                {
                    "r0": int(r0 + hr0),
                    "c0": int(c0 + hc0),
                    "mask": hole_mask,
                }
            )
        if holes:
            frames.append({"color": color, "holes": holes})
    return frames


def _piece_shape_counts(grid: np.ndarray) -> Counter:
    counts = Counter()
    for coords in _components(grid == PIECE_COLOR):
        counts[_shape_key(_mask_from_coords(coords))] += 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match grey fragments to enclosed black holes inside colored rectangular frames.
    Fill each hole with the matching fragment's shape under rotation, recolored to the
    frame's border color, and erase the fragment from its original position.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[g == PIECE_COLOR] = BG

    piece_counts = _piece_shape_counts(g)
    for frame in _find_frames(g):
        color = int(frame["color"])
        for hole in frame["holes"]:
            key = _shape_key(hole["mask"])
            if piece_counts[key] <= 0:
                continue
            piece_counts[key] -= 1
            patch = out[
                hole["r0"] : hole["r0"] + hole["mask"].shape[0],
                hole["c0"] : hole["c0"] + hole["mask"].shape[1],
            ]
            patch[hole["mask"]] = color
    return out


def _sample_hole_counts(n_frames: int) -> list[int]:
    total = random.randint(max(2, n_frames), min(TOTAL_HOLES_MAX, n_frames * MAX_HOLES_PER_FRAME))
    counts = [1] * n_frames
    remaining = total - n_frames
    while remaining > 0:
        choices = [i for i, v in enumerate(counts) if v < MAX_HOLES_PER_FRAME]
        idx = random.choice(choices)
        counts[idx] += 1
        remaining -= 1
    random.shuffle(counts)
    return counts


def _random_polyomino() -> np.ndarray | None:
    for _ in range(200):
        target = random.randint(*POLYOMINO_AREA_RANGE)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = []
            weights = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in cells:
                        continue
                    frontier.append((nr, nc))
                    support = sum((nr + ddr, nc + ddc) in cells for ddr, ddc in DIR4)
                    weights.append(support)
            cells.add(random.choices(frontier, weights=weights, k=1)[0])
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0 = min(rs), min(cs)
        shifted = {(r - r0, c - c0) for r, c in cells}
        h = max(r for r, _ in shifted) + 1
        w = max(c for _, c in shifted) + 1
        if h > POLYOMINO_BOX_MAX or w > POLYOMINO_BOX_MAX:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in shifted:
            mask[r, c] = True
        if int(mask.sum()) == int(h * w):
            continue
        if len(_rotation_variants(mask)) == 1 and random.random() < 0.8:
            continue
        return mask
    return None


def _generate_unique_shape(used_keys: set) -> np.ndarray | None:
    for _ in range(200):
        mask = _random_polyomino()
        if mask is None:
            continue
        key = _shape_key(mask)
        if key in used_keys:
            continue
        return mask
    return None


def _can_place_hole(occupied: np.ndarray, mask: np.ndarray, top: int, left: int, pad: int = 1) -> bool:
    coords = np.argwhere(mask)
    for rr, cc in coords:
        r = int(top + rr)
        c = int(left + cc)
        r0 = max(0, r - pad)
        r1 = min(occupied.shape[0], r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(occupied.shape[1], c + pad + 1)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _pack_frame(color: int, base_masks: list[np.ndarray]) -> dict | None:
    oriented = [_trim(np.rot90(mask, random.randrange(4))) for mask in base_masks]
    area = sum(int(mask.sum()) for mask in oriented)
    max_h = max(int(mask.shape[0]) for mask in oriented)
    max_w = max(int(mask.shape[1]) for mask in oriented)

    for _ in range(FRAME_PACK_TRIES):
        target_side = int(np.ceil(np.sqrt(area * 2.4)))
        inner_h = max(max_h + 2, target_side + random.randint(0, 2), 6)
        inner_w = max(max_w + 2, target_side + random.randint(0, 3), 6)
        inner_h = min(inner_h, 11)
        inner_w = min(inner_w, 11)

        occupied = np.zeros((inner_h, inner_w), dtype=bool)
        placements = [None] * len(oriented)
        order = sorted(range(len(oriented)), key=lambda i: int(oriented[i].sum()), reverse=True)
        ok = True
        for idx in order:
            mask = oriented[idx]
            h, w = map(int, mask.shape)
            candidates = []
            for top in range(1, inner_h - h):
                for left in range(1, inner_w - w):
                    if _can_place_hole(occupied, mask, top, left, pad=1):
                        candidates.append((top, left))
            if not candidates:
                ok = False
                break
            top, left = random.choice(candidates)
            occupied[top : top + h, left : left + w] |= mask
            placements[idx] = (top, left)
        if not ok:
            continue

        patch = np.full((inner_h + 2, inner_w + 2), color, dtype=int)
        holes = []
        for mask, (top, left) in zip(oriented, placements):
            r0 = top + 1
            c0 = left + 1
            sub = patch[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
            sub[mask] = BG
            holes.append({"mask": mask, "r0": r0, "c0": c0})
        return {"color": color, "patch": patch, "holes": holes}
    return None


def _bbox_is_free(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    r0 = max(0, top - pad)
    r1 = min(occupied.shape[0], top + h + pad)
    c0 = max(0, left - pad)
    c1 = min(occupied.shape[1], left + w + pad)
    return not occupied[r0:r1, c0:c1].any()


def _place_bbox(occupied: np.ndarray, h: int, w: int, pad: int, tries: int) -> tuple[int, int] | None:
    margin = 1
    max_top = occupied.shape[0] - h - margin
    max_left = occupied.shape[1] - w - margin
    if max_top < margin or max_left < margin:
        return None
    for _ in range(tries):
        top = random.randint(margin, max_top)
        left = random.randint(margin, max_left)
        if _bbox_is_free(occupied, top, left, h, w, pad):
            occupied[top : top + h, left : left + w] = True
            return top, left
    return None


def _build_global_holes(
    frame_specs: list[dict], grid_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    grid = np.zeros(grid_shape, dtype=int)
    expected = np.zeros(grid_shape, dtype=int)
    occupied = np.zeros(grid_shape, dtype=bool)
    global_holes = []

    ordered_frames = frame_specs[:]
    random.shuffle(ordered_frames)
    ordered_frames.sort(key=lambda rec: -int(rec["patch"].shape[0] * rec["patch"].shape[1]))
    for frame in ordered_frames:
        patch = frame["patch"]
        pos = _place_bbox(
            occupied, int(patch.shape[0]), int(patch.shape[1]), pad=2, tries=GLOBAL_FRAME_PLACE_TRIES
        )
        if pos is None:
            return None
        top, left = pos
        grid[top : top + patch.shape[0], left : left + patch.shape[1]] = patch
        expected[top : top + patch.shape[0], left : left + patch.shape[1]] = patch
        for hole in frame["holes"]:
            global_holes.append(
                {
                    "color": int(frame["color"]),
                    "mask": hole["mask"],
                    "r0": int(top + hole["r0"]),
                    "c0": int(left + hole["c0"]),
                }
            )
    return grid, expected, global_holes


def _piece_mask_for_hole(mask: np.ndarray, force_change: bool) -> np.ndarray | None:
    options = []
    for k in range(4):
        rot = _trim(np.rot90(mask, k))
        if not force_change or not np.array_equal(rot, mask):
            options.append(rot)
    if not options:
        return None
    return random.choice(options)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_frames = random.choice(FRAME_COUNT_OPTIONS)
    hole_counts = _sample_hole_counts(n_frames)
    frame_colors = random.sample(FRAME_COLORS, n_frames)

    used_keys = set()
    frame_specs = []
    has_rotationally_distinct_shape = False
    for color, hole_count in zip(frame_colors, hole_counts):
        base_masks = []
        for _ in range(hole_count):
            mask = _generate_unique_shape(used_keys)
            if mask is None:
                return None
            used_keys.add(_shape_key(mask))
            base_masks.append(mask)
            if len(_rotation_variants(mask)) > 1:
                has_rotationally_distinct_shape = True
        frame_spec = _pack_frame(color, base_masks)
        if frame_spec is None:
            return None
        frame_specs.append(frame_spec)
    if not has_rotationally_distinct_shape:
        return None

    built = _build_global_holes(frame_specs, (h, w))
    if built is None:
        return None
    grid, expected, global_holes = built

    asymmetric = [i for i, hole in enumerate(global_holes) if len(_rotation_variants(hole["mask"])) > 1]
    if not asymmetric:
        return None
    forced_idx = random.choice(asymmetric)

    piece_specs = []
    for i, hole in enumerate(global_holes):
        piece_mask = _piece_mask_for_hole(hole["mask"], force_change=(i == forced_idx))
        if piece_mask is None:
            return None
        piece_specs.append({"mask": piece_mask, "area": int(piece_mask.sum())})
        patch = expected[
            hole["r0"] : hole["r0"] + hole["mask"].shape[0],
            hole["c0"] : hole["c0"] + hole["mask"].shape[1],
        ]
        patch[hole["mask"]] = int(hole["color"])

    occupied = expected != BG
    random.shuffle(piece_specs)
    piece_specs.sort(key=lambda rec: -int(rec["area"]))
    for piece in piece_specs:
        mask = piece["mask"]
        pos = _place_bbox(
            occupied, int(mask.shape[0]), int(mask.shape[1]), pad=1, tries=GLOBAL_PIECE_PLACE_TRIES
        )
        if pos is None:
            return None
        top, left = pos
        patch = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
        patch[mask] = PIECE_COLOR

    if np.array_equal(grid, expected) or not np.any(grid == PIECE_COLOR):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
