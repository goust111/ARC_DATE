from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import (
    Color,
    bbox_coords,
    components_mask,
    find_components,
    first_free_location,
    get_bg,
    in_bounds,
    neighbors4,
    paste,
)


BG = int(Color.BLACK)
WALL = int(Color.GREY)
PIECE_COLORS = tuple(color for color in Color.NOT_BLACK if int(color) != WALL)

MAX_GENERATION_ATTEMPTS = 400
WALL_SIZE_RANGE = (6, 10)
CANVAS_SIZE_RANGE = (20, 28)
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _mask_key(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return (
        int(mask.shape[0]),
        int(mask.shape[1]),
        tuple(int(v) for v in mask.astype(np.uint8).ravel()),
    )


def _sorted_components(components: list[dict]) -> list[dict]:
    return sorted(components, key=lambda comp: tuple(map(int, comp["bbox"])))


def _wall_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _enclosed_regions(barrier_mask: np.ndarray) -> list[np.ndarray]:
    coords = np.argwhere(barrier_mask)
    if coords.size == 0:
        return []

    r0, c0, r1, c1 = bbox_coords(coords)
    r0 = max(0, r0 - 1)
    c0 = max(0, c0 - 1)
    r1 = min(barrier_mask.shape[0] - 1, r1 + 1)
    c1 = min(barrier_mask.shape[1] - 1, c1 + 1)

    local_barrier = barrier_mask[r0 : r1 + 1, c0 : c1 + 1]
    free = ~local_barrier
    regions: list[np.ndarray] = []
    for comp in components_mask(free, connectivity=4):
        rs = comp[:, 0]
        cs = comp[:, 1]
        touches_border = bool(
            np.any(rs == 0)
            or np.any(rs == free.shape[0] - 1)
            or np.any(cs == 0)
            or np.any(cs == free.shape[1] - 1)
        )
        if not touches_border:
            regions.append(comp + np.array([r0, c0]))
    return regions


def _piece_variants(grid: np.ndarray) -> list[dict]:
    piece_colors = [int(color) for color in np.unique(grid) if int(color) not in {BG, WALL}]
    if not piece_colors:
        return []

    pieces = _sorted_components(
        find_components(
            grid,
            bg=BG,
            colors=piece_colors,
            connectivity=4,
            monochromatic=True,
            return_format="dict",
        )
    )
    out = []
    for index, piece in enumerate(pieces):
        coords = np.asarray(piece["coords"], dtype=int)
        shape, _bbox = _mask_from_coords(coords)
        local_coords = np.argwhere(shape)
        sprite = np.full(shape.shape, BG, dtype=int)
        sprite[shape] = int(piece["color"])
        out.append(
            {
                "index": index,
                "coords": coords,
                "bbox": tuple(map(int, piece["bbox"])),
                "color": int(piece["color"]),
                "shape": shape,
                "local_coords": local_coords,
                "sprite": sprite,
            }
        )
    return out


def _wall_variants(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    walls = _sorted_components(
        find_components(
            grid,
            bg=BG,
            colors=[WALL],
            connectivity=4,
            monochromatic=True,
            return_format="dict",
        )
    )

    wall_labels = np.full(grid.shape, -1, dtype=int)
    enriched = []
    for index, wall in enumerate(walls):
        coords = np.asarray(wall["coords"], dtype=int)
        wall_labels[coords[:, 0], coords[:, 1]] = index
        mask = _wall_mask(grid.shape, coords)
        enriched.append(
            {
                "index": index,
                "coords": coords,
                "bbox": tuple(map(int, wall["bbox"])),
                "mask": mask,
                "before_regions": _enclosed_regions(mask),
            }
        )
    return enriched, wall_labels


def _find_piece_placement(
    grid: np.ndarray,
    piece: dict,
    walls: list[dict],
    wall_labels: np.ndarray,
) -> dict | None:
    base = np.array(grid, copy=True)
    base[piece["coords"][:, 0], piece["coords"][:, 1]] = BG

    candidates = []
    H, W = base.shape
    h, w = piece["shape"].shape
    local_coords = piece["local_coords"]

    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            placed = local_coords + np.array([r0, c0])
            if np.any(base[placed[:, 0], placed[:, 1]] != BG):
                continue

            touched_walls = set()
            for r, c in placed:
                for nr, nc in neighbors4(int(r), int(c)):
                    if in_bounds(base.shape, nr, nc) and wall_labels[nr, nc] >= 0:
                        touched_walls.add(int(wall_labels[nr, nc]))
            if len(touched_walls) != 1:
                continue

            wall_id = next(iter(touched_walls))
            wall = walls[wall_id]
            if wall["before_regions"]:
                continue

            barrier = np.array(wall["mask"], copy=True)
            barrier[placed[:, 0], placed[:, 1]] = True
            after_regions = _enclosed_regions(barrier)
            if not after_regions:
                continue

            fill = np.concatenate(after_regions, axis=0)
            candidates.append(
                {
                    "piece_index": int(piece["index"]),
                    "wall_index": wall_id,
                    "top_left": (int(r0), int(c0)),
                    "fill": fill,
                    "fill_area": int(fill.shape[0]),
                }
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda cand: (
            -cand["fill_area"],
            cand["wall_index"],
            cand["top_left"][0],
            cand["top_left"][1],
        )
    )
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock each colored fragment into its exactly matching wall gap, then flood only the region sealed by that docking with the fragment color.

    Input structure:
    - A black background contains several grey 4-connected wall loops; some are already closed and some are open because a contiguous border segment is missing.
    - Separate monochrome colored fragments sit elsewhere on the canvas and may translate to one wall gap without rotating.
    - Open walls expose black interiors to the outside until a matching fragment seals their missing border cells.

    Rule:
    - Treat each colored fragment as a rigid shape and search for a translation onto black cells where it touches exactly one grey wall component.
    - Accept a placement only when that fragment changes the touched wall from having no enclosed region to having at least one enclosed region.
    - Remove every accepted fragment from its original location, paste it into the sealing gap, and flood the newly enclosed region with that fragment's color; leave unmatched fragments, unmatched gaps, and originally closed walls unchanged.

    Key insight:
    - The output color comes from the fragment that triggers the sealing event, so the task is to detect closure caused by a moved piece rather than to statically fill every loop interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(get_bg(g)) == BG

    walls, wall_labels = _wall_variants(g)
    pieces = _piece_variants(g)

    placements = []
    for piece in pieces:
        placement = _find_piece_placement(g, piece, walls, wall_labels)
        if placement is not None:
            placements.append(placement)

    placements.sort(
        key=lambda cand: (
            -cand["fill_area"],
            cand["wall_index"],
            cand["piece_index"],
        )
    )

    out = np.array(g, copy=True)
    used_pieces: set[int] = set()
    used_walls: set[int] = set()
    piece_by_index = {int(piece["index"]): piece for piece in pieces}

    for placement in placements:
        piece_index = int(placement["piece_index"])
        wall_index = int(placement["wall_index"])
        if piece_index in used_pieces or wall_index in used_walls:
            continue

        piece = piece_by_index[piece_index]
        out[piece["coords"][:, 0], piece["coords"][:, 1]] = BG
        paste(
            out,
            piece["sprite"],
            int(placement["top_left"][0]),
            int(placement["top_left"][1]),
            transparent=BG,
        )
        out[placement["fill"][:, 0], placement["fill"][:, 1]] = int(piece["color"])
        used_pieces.add(piece_index)
        used_walls.add(wall_index)

    return out


def _gap_option(coords: list[tuple[int, int]]) -> dict:
    arr = np.array(sorted(coords), dtype=int)
    mask, box = _mask_from_coords(arr)
    return {
        "coords": arr,
        "bbox": box,
        "mask": mask,
        "key": _mask_key(mask),
    }


def _gap_options(h: int, w: int) -> list[dict]:
    options = []

    max_hseg = min(3, w - 2)
    for row in (0, h - 1):
        for length in range(1, max_hseg + 1):
            for start in range(1, w - length):
                options.append(_gap_option([(row, c) for c in range(start, start + length)]))

    max_vseg = min(3, h - 2)
    for col in (0, w - 1):
        for length in range(1, max_vseg + 1):
            for start in range(1, h - length):
                options.append(_gap_option([(r, col) for r in range(start, start + length)]))

    max_arm_r = min(3, h - 2)
    max_arm_c = min(3, w - 2)
    if max_arm_r >= 2 and max_arm_c >= 2:
        for arm_r in range(2, max_arm_r + 1):
            for arm_c in range(2, max_arm_c + 1):
                options.extend(
                    [
                        _gap_option(
                            [(0, c) for c in range(arm_c)] + [(r, 0) for r in range(arm_r)]
                        ),
                        _gap_option(
                            [(0, w - 1 - c) for c in range(arm_c)]
                            + [(r, w - 1) for r in range(arm_r)]
                        ),
                        _gap_option(
                            [(h - 1, c) for c in range(arm_c)]
                            + [(h - 1 - r, 0) for r in range(arm_r)]
                        ),
                        _gap_option(
                            [(h - 1, w - 1 - c) for c in range(arm_c)]
                            + [(h - 1 - r, w - 1) for r in range(arm_r)]
                        ),
                    ]
                )
    return options


def _wall_sprite(h: int, w: int, gap: dict | None) -> np.ndarray:
    sprite = np.zeros((h, w), dtype=int)
    sprite[0, :] = WALL
    sprite[-1, :] = WALL
    sprite[:, 0] = WALL
    sprite[:, -1] = WALL
    if gap is not None:
        sprite[gap["coords"][:, 0], gap["coords"][:, 1]] = BG
    return sprite


def _random_piece_mask(forbidden_keys: set[tuple[int, int, tuple[int, ...]]]) -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        target = random.randint(4, min(6, h * w))
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        arr = np.array(sorted(cells), dtype=int)
        mask, _box = _mask_from_coords(arr)
        key = _mask_key(mask)
        if key in forbidden_keys:
            continue
        if mask.shape[0] == 1 or mask.shape[1] == 1:
            continue
        if int(mask.sum()) < 4:
            continue
        return mask
    return None


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.full(mask.shape, BG, dtype=int)
    sprite[mask] = int(color)
    return sprite


def _place_sprite(canvas: np.ndarray, sprite: np.ndarray) -> tuple[int, int] | None:
    return first_free_location(
        canvas,
        sprite,
        transparent=BG,
        border=PLACEMENT_BORDER,
        padding=PLACEMENT_PADDING,
        padding_connectivity=8,
    )


def _expected_output(grid: np.ndarray, walls: list[dict], pieces: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)

    for piece in pieces:
        if piece["kind"] != "matched":
            continue
        pr, pc = piece["origin"]
        mask = piece["sprite"] != BG
        out[pr : pr + piece["sprite"].shape[0], pc : pc + piece["sprite"].shape[1]][mask] = BG

    for wall in walls:
        if wall["kind"] != "matched":
            continue
        wr, wc = wall["origin"]
        gap = wall["gap"]
        color = int(wall["color"])
        paste(out, _colorize(gap["mask"], color), wr + gap["bbox"][0], wc + gap["bbox"][1], transparent=BG)
        out[wr + 1 : wr + wall["h"] - 1, wc + 1 : wc + wall["w"] - 1] = color

    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    H = random.randint(*CANVAS_SIZE_RANGE)
    W = random.randint(*CANVAS_SIZE_RANGE)
    grid = np.full((H, W), BG, dtype=int)

    matched_count = random.choice([1, 2, 2])
    unmatched_open_count = int(random.random() < 0.65)
    closed_count = int(random.random() < 0.65)
    extra_piece_count = int(random.random() < 0.65)
    if unmatched_open_count + closed_count + extra_piece_count == 0:
        choice = random.choice(["open", "closed", "extra"])
        unmatched_open_count += int(choice == "open")
        closed_count += int(choice == "closed")
        extra_piece_count += int(choice == "extra")

    wall_kinds = (
        ["matched"] * matched_count
        + ["open"] * unmatched_open_count
        + ["closed"] * closed_count
    )
    random.shuffle(wall_kinds)

    used_gap_keys: set[tuple[int, int, tuple[int, ...]]] = set()
    walls = []
    for kind in wall_kinds:
        built = None
        for _ in range(80):
            h = random.randint(*WALL_SIZE_RANGE)
            w = random.randint(*WALL_SIZE_RANGE)
            gap = None
            if kind != "closed":
                candidates = [opt for opt in _gap_options(h, w) if opt["key"] not in used_gap_keys]
                if not candidates:
                    continue
                gap = random.choice(candidates)
            built = {
                "kind": kind,
                "h": h,
                "w": w,
                "gap": gap,
                "sprite": _wall_sprite(h, w, gap),
            }
            if gap is not None:
                used_gap_keys.add(gap["key"])
            break
        if built is None:
            return None
        walls.append(built)

    for wall in sorted(walls, key=lambda item: item["h"] * item["w"], reverse=True):
        loc = _place_sprite(grid, wall["sprite"])
        if loc is None:
            return None
        wall["origin"] = (int(loc[0]), int(loc[1]))
        paste(grid, wall["sprite"], loc[0], loc[1], transparent=BG)

    total_piece_count = matched_count + extra_piece_count
    colors = random.sample(PIECE_COLORS, total_piece_count)
    pieces = []
    color_iter = iter(colors)
    used_piece_keys = set(used_gap_keys)

    for wall in walls:
        if wall["kind"] != "matched":
            continue
        color = int(next(color_iter))
        wall["color"] = color
        piece_mask = wall["gap"]["mask"]
        piece = {
            "kind": "matched",
            "wall": wall,
            "sprite": _colorize(piece_mask, color),
            "shape_key": wall["gap"]["key"],
            "color": color,
        }
        pieces.append(piece)

    for _ in range(extra_piece_count):
        color = int(next(color_iter))
        mask = _random_piece_mask(used_piece_keys)
        if mask is None:
            return None
        key = _mask_key(mask)
        used_piece_keys.add(key)
        pieces.append(
            {
                "kind": "extra",
                "sprite": _colorize(mask, color),
                "shape_key": key,
                "color": color,
            }
        )

    random.shuffle(pieces)
    for piece in pieces:
        loc = _place_sprite(grid, piece["sprite"])
        if loc is None:
            return None
        piece["origin"] = (int(loc[0]), int(loc[1]))
        paste(grid, piece["sprite"], loc[0], loc[1], transparent=BG)

    expected = _expected_output(grid, walls, pieces)
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            grid, _expected = case
            return grid
    raise RuntimeError("failed to generate a valid sealing-and-fill ARC input")
