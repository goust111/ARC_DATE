from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, bbox_h, bbox_w, find_components, get_bg


GEN_MAX_ATTEMPTS = 600
GRID_MIN = 20
GRID_MAX = 24
FRAME_SIZE_RANGE = (6, 8)
CORNERS = ("ul", "ur", "ll", "lr")
OPPOSITE = {"ul": "lr", "ur": "ll", "ll": "ur", "lr": "ul"}
SPRITE_PATTERNS = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
]


def _frame_border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _interior_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame needs a nonempty interior"
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _coords_inside_bbox(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(
        np.all((coords[:, 0] >= r0) & (coords[:, 0] <= r1) & (coords[:, 1] >= c0) & (coords[:, 1] <= c1))
    )


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _color_frame_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(grid == int(color))
    if coords.size == 0:
        return None

    bbox = bbox_coords(coords)
    if bbox_h(bbox) < 4 or bbox_w(bbox) < 4:
        return None

    r0, c0, r1, c1 = bbox
    mask = np.zeros_like(grid, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True

    if np.any(mask[r0 + 1 : r1, c0 + 1 : c1]):
        return None
    if not mask[r0, c0 : c1 + 1].any():
        return None
    if not mask[r1, c0 : c1 + 1].any():
        return None
    if not mask[r0 : r1 + 1, c0].any():
        return None
    if not mask[r0 : r1 + 1, c1].any():
        return None
    return bbox


def _frame_infos(grid: np.ndarray, bg: int) -> list[dict]:
    infos: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != int(bg)):
        bbox = _color_frame_bbox(grid, color)
        if bbox is None:
            continue
        infos.append(
            {
                "color": int(color),
                "bbox": bbox,
                "interior_bbox": _interior_bbox(bbox),
                "border_mask": _frame_border_mask(grid.shape, bbox),
            }
        )
    assert infos, "expected at least one frame color"
    return infos


def _anchor_scores(
    sprite_bbox: tuple[int, int, int, int], interior_bbox: tuple[int, int, int, int]
) -> dict[str, int]:
    sr0, sc0, sr1, sc1 = sprite_bbox
    ir0, ic0, ir1, ic1 = interior_bbox
    return {
        "ul": int(sr0 - ir0 + sc0 - ic0),
        "ur": int(sr0 - ir0 + ic1 - sc1),
        "ll": int(ir1 - sr1 + sc0 - ic0),
        "lr": int(ir1 - sr1 + ic1 - sc1),
    }


def _unique_anchor(
    sprite_bbox: tuple[int, int, int, int], interior_bbox: tuple[int, int, int, int]
) -> str:
    scores = _anchor_scores(sprite_bbox, interior_bbox)
    best = min(scores.values())
    anchors = [anchor for anchor, score in scores.items() if score == best]
    assert len(anchors) == 1, "sprite anchor must be unique"
    return anchors[0]


def _translated_coords(
    coords: np.ndarray,
    sprite_bbox: tuple[int, int, int, int],
    old_interior: tuple[int, int, int, int],
    new_interior: tuple[int, int, int, int],
) -> np.ndarray:
    anchor = _unique_anchor(sprite_bbox, old_interior)
    sr0, sc0, sr1, sc1 = sprite_bbox
    ir0, ic0, ir1, ic1 = old_interior
    nr0, nc0, nr1, nc1 = new_interior
    h = bbox_h(sprite_bbox)
    w = bbox_w(sprite_bbox)

    if anchor == "ul":
        new_r0 = nr0 + (sr0 - ir0)
        new_c0 = nc0 + (sc0 - ic0)
    elif anchor == "ur":
        new_r0 = nr0 + (sr0 - ir0)
        new_c0 = nc1 - (ic1 - sc1) - w + 1
    elif anchor == "ll":
        new_r0 = nr1 - (ir1 - sr1) - h + 1
        new_c0 = nc0 + (sc0 - ic0)
    else:
        new_r0 = nr1 - (ir1 - sr1) - h + 1
        new_c0 = nc1 - (ic1 - sc1) - w + 1

    delta = np.array([new_r0 - sr0, new_c0 - sc0], dtype=int)
    moved = coords + delta
    assert _coords_inside_bbox(moved, new_interior), "moved sprite must fit in destination interior"
    return moved


def _assign_frame_sprites(grid: np.ndarray, bg: int, frames: list[dict]) -> tuple[dict[int, dict], list[dict]]:
    frame_colors = {frame["color"] for frame in frames}
    sprite_colors = [int(c) for c in np.unique(grid) if int(c) not in frame_colors | {int(bg)}]
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=sprite_colors,
    )

    assignments: dict[int, list[dict]] = {index: [] for index in range(len(frames))}
    extras: list[dict] = []
    for comp in components:
        owners = [
            index
            for index, frame in enumerate(frames)
            if _coords_inside_bbox(comp["coords"], frame["interior_bbox"])
        ]
        if len(owners) == 1:
            assignments[owners[0]].append(comp)
        else:
            extras.append(comp)

    sprite_by_frame: dict[int, dict] = {}
    for index, comps in assignments.items():
        assert len(comps) == 1, "expected exactly one sprite inside each frame"
        sprite_by_frame[index] = comps[0]
    return sprite_by_frame, extras


def _transfer_edges(grid: np.ndarray, frames: list[dict]) -> dict[int, int]:
    edges: dict[int, int] = {}
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            inter = frames[i]["border_mask"] & frames[j]["border_mask"]
            if not inter.any():
                continue
            visible = {int(v) for v in np.unique(grid[inter])}
            ci = frames[i]["color"]
            cj = frames[j]["color"]
            if cj in visible and ci not in visible:
                assert i not in edges, "each donor frame must have a unique receiver"
                edges[i] = j
            elif ci in visible and cj not in visible:
                assert j not in edges, "each donor frame must have a unique receiver"
                edges[j] = i
            else:
                raise AssertionError("frame overlap visibility must pick exactly one top frame")
    return edges


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Move each buried frame's sprite into the frame that visibly covers the crossing.

    Input structure:
    - Multiple one-pixel-thick rectangular frames overlap partially on a uniform background.
    - Each frame contains one monochrome sprite whose color is distinct from all frame colors.
    - Additional isolated frames and isolated sprites may appear as distractors.

    Rule:
    - Reconstruct every frame from the bbox of its visible color.
    - At each frame-border crossing, the visible color marks the upper frame.
    - If frame A is hidden under frame B, move A's inner sprite into B's interior.
    - The moved sprite keeps the same offset from its uniquely closest interior corner.
    - All transfers happen simultaneously; frame borders stay unchanged.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(grid))
    frames = _frame_infos(grid, bg)
    sprites, _extras = _assign_frame_sprites(grid, bg, frames)
    edges = _transfer_edges(grid, frames)

    out = grid.copy()
    pending_moves: list[tuple[np.ndarray, int]] = []
    for donor, receiver in edges.items():
        sprite = sprites[donor]
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = bg
        pending_moves.append(
            (
                _translated_coords(
                    sprite["coords"],
                    sprite["bbox"],
                    frames[donor]["interior_bbox"],
                    frames[receiver]["interior_bbox"],
                ),
                int(sprite["color"]),
            )
        )

    for coords, color in pending_moves:
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _make_bbox(r0: int, c0: int, h: int, w: int) -> tuple[int, int, int, int]:
    return r0, c0, r0 + h - 1, c0 + w - 1


def _place_relative_box(
    parent_bbox: tuple[int, int, int, int],
    h: int,
    w: int,
    corner: str,
    shape: tuple[int, int],
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    ph = bbox_h(parent_bbox)
    pw = bbox_w(parent_bbox)
    max_ovh = min(h - 2, ph - 2, 4)
    max_ovw = min(w - 2, pw - 2, 4)
    if max_ovh < 2 or max_ovw < 2:
        return None

    ovh = rng.randint(2, max_ovh)
    ovw = rng.randint(2, max_ovw)
    pr0, pc0, pr1, pc1 = parent_bbox
    if corner == "ul":
        r0 = pr0 - (h - ovh)
        c0 = pc0 - (w - ovw)
    elif corner == "ur":
        r0 = pr0 - (h - ovh)
        c0 = pc1 - ovw + 1
    elif corner == "ll":
        r0 = pr1 - ovh + 1
        c0 = pc0 - (w - ovw)
    else:
        r0 = pr1 - ovh + 1
        c0 = pc1 - ovw + 1

    bbox = _make_bbox(r0, c0, h, w)
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] >= shape[0] or bbox[3] >= shape[1]:
        return None
    return bbox


def _origin_from_anchor(
    interior_bbox: tuple[int, int, int, int],
    patch_shape: tuple[int, int],
    anchor: str,
    row_gap: int,
    col_gap: int,
) -> tuple[int, int]:
    ir0, ic0, ir1, ic1 = interior_bbox
    ph, pw = patch_shape
    if anchor == "ul":
        return ir0 + row_gap, ic0 + col_gap
    if anchor == "ur":
        return ir0 + row_gap, ic1 - pw + 1 - col_gap
    if anchor == "ll":
        return ir1 - ph + 1 - row_gap, ic0 + col_gap
    return ir1 - ph + 1 - row_gap, ic1 - pw + 1 - col_gap


def _place_sprite(
    frame_bbox: tuple[int, int, int, int],
    pattern: np.ndarray,
    anchor: str,
    color: int,
    rng: random.Random,
) -> dict | None:
    interior = _interior_bbox(frame_bbox)
    ih = bbox_h(interior)
    iw = bbox_w(interior)
    ph, pw = map(int, pattern.shape)
    if ph > ih or pw > iw:
        return None

    max_row_gap = min(1, ih - ph)
    max_col_gap = min(1, iw - pw)
    for _ in range(40):
        row_gap = rng.randint(0, max_row_gap)
        col_gap = rng.randint(0, max_col_gap)
        r0, c0 = _origin_from_anchor(interior, pattern.shape, anchor, row_gap, col_gap)
        coords = np.argwhere(pattern) + np.array([r0, c0], dtype=int)
        sprite_bbox = bbox_coords(coords)
        scores = _anchor_scores(sprite_bbox, interior)
        best = min(scores.values())
        anchors = [name for name, score in scores.items() if score == best]
        if anchors != [anchor]:
            continue
        return {
            "color": int(color),
            "pattern": pattern.copy(),
            "coords": coords,
            "bbox": sprite_bbox,
        }
    return None


def _paint_sprite(grid: np.ndarray, sprite: dict) -> None:
    grid[sprite["coords"][:, 0], sprite["coords"][:, 1]] = int(sprite["color"])


def _available_corners(used: set[str]) -> list[str]:
    return [corner for corner in CORNERS if corner not in used]


def _build_active_frames(
    shape: tuple[int, int], active_count: int, colors: list[int], rng: random.Random
) -> list[dict] | None:
    h0 = rng.randint(*FRAME_SIZE_RANGE)
    w0 = rng.randint(*FRAME_SIZE_RANGE)
    margin = 5
    if shape[0] - h0 - 2 * margin < 0 or shape[1] - w0 - 2 * margin < 0:
        return None

    root_r0 = rng.randint(margin, shape[0] - h0 - margin)
    root_c0 = rng.randint(margin, shape[1] - w0 - margin)
    root_bbox = _make_bbox(root_r0, root_c0, h0, w0)

    if active_count == 2:
        topology = "star"
    elif active_count == 3:
        topology = rng.choice(["star", "chain"])
    else:
        topology = "star"

    if topology == "star":
        child_corners = rng.sample(list(CORNERS), active_count - 1)
        root_anchor = _available_corners(set(child_corners))[0]
        frames = [
            {
                "bbox": root_bbox,
                "color": int(colors[0]),
                "parent": None,
                "anchor": root_anchor,
                "z": 1,
            }
        ]
        for index, corner in enumerate(child_corners, start=1):
            child_bbox = _place_relative_box(
                root_bbox,
                rng.randint(*FRAME_SIZE_RANGE),
                rng.randint(*FRAME_SIZE_RANGE),
                corner,
                shape,
                rng,
            )
            if child_bbox is None:
                return None
            frames.append(
                {
                    "bbox": child_bbox,
                    "color": int(colors[index]),
                    "parent": 0,
                    "anchor": corner,
                    "z": 0,
                }
            )
    else:
        low_corner = rng.choice(list(CORNERS))
        high_corner = OPPOSITE[low_corner]
        high_anchor_choices = _available_corners({high_corner})
        frames = [
            {
                "bbox": None,
                "color": int(colors[0]),
                "parent": 1,
                "anchor": low_corner,
                "z": 0,
            },
            {
                "bbox": root_bbox,
                "color": int(colors[1]),
                "parent": 2,
                "anchor": high_corner,
                "z": 1,
            },
            {
                "bbox": None,
                "color": int(colors[2]),
                "parent": None,
                "anchor": rng.choice(high_anchor_choices),
                "z": 2,
            },
        ]
        low_bbox = _place_relative_box(
            root_bbox,
            rng.randint(*FRAME_SIZE_RANGE),
            rng.randint(*FRAME_SIZE_RANGE),
            low_corner,
            shape,
            rng,
        )
        high_bbox = _place_relative_box(
            root_bbox,
            rng.randint(*FRAME_SIZE_RANGE),
            rng.randint(*FRAME_SIZE_RANGE),
            high_corner,
            shape,
            rng,
        )
        if low_bbox is None or high_bbox is None:
            return None
        frames[0]["bbox"] = low_bbox
        frames[2]["bbox"] = high_bbox

    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            allowed = frames[i]["parent"] == j or frames[j]["parent"] == i
            if _bbox_intersects(frames[i]["bbox"], frames[j]["bbox"]) != allowed:
                return None
    return frames


def _add_isolated_frames(
    frames: list[dict], shape: tuple[int, int], colors: list[int], rng: random.Random
) -> bool:
    for color in colors:
        placed = False
        for _ in range(80):
            h = rng.randint(*FRAME_SIZE_RANGE)
            w = rng.randint(*FRAME_SIZE_RANGE)
            r0 = rng.randint(0, shape[0] - h)
            c0 = rng.randint(0, shape[1] - w)
            bbox = _make_bbox(r0, c0, h, w)
            if any(_bbox_intersects(bbox, frame["bbox"]) for frame in frames):
                continue
            frames.append(
                {
                    "bbox": bbox,
                    "color": int(color),
                    "parent": None,
                    "anchor": rng.choice(list(CORNERS)),
                    "z": 0,
                }
            )
            placed = True
            break
        if not placed:
            return False
    return True


def _place_standalone_sprites(
    grid: np.ndarray, frames: list[dict], sprite_colors: list[int], rng: random.Random
) -> list[dict] | None:
    standalone: list[dict] = []
    count = rng.randint(0, 2)
    frame_boxes = [frame["bbox"] for frame in frames]
    for _ in range(count):
        pattern = rng.choice(SPRITE_PATTERNS)
        ph, pw = map(int, pattern.shape)
        color = int(rng.choice(sprite_colors))
        placed = False
        for _ in range(80):
            r0 = rng.randint(0, grid.shape[0] - ph)
            c0 = rng.randint(0, grid.shape[1] - pw)
            coords = np.argwhere(pattern) + np.array([r0, c0], dtype=int)
            bbox = bbox_coords(coords)
            if any(_bbox_intersects(bbox, frame_bbox) for frame_bbox in frame_boxes):
                continue
            if np.any(grid[coords[:, 0], coords[:, 1]] != get_bg(grid)):
                continue
            sprite = {"color": color, "pattern": pattern.copy(), "coords": coords, "bbox": bbox}
            _paint_sprite(grid, sprite)
            standalone.append(sprite)
            placed = True
            break
        if not placed:
            return None
    return standalone


def _expected_output(input_grid: np.ndarray, frames: list[dict], bg: int) -> np.ndarray | None:
    out = input_grid.copy()
    moves: list[tuple[np.ndarray, int]] = []
    for index, frame in enumerate(frames):
        if frame["parent"] is None:
            continue
        sprite = frame["sprite"]
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = int(bg)
        moved = _translated_coords(
            sprite["coords"],
            sprite["bbox"],
            _interior_bbox(frame["bbox"]),
            _interior_bbox(frames[frame["parent"]]["bbox"]),
        )
        moves.append((moved, int(sprite["color"])))

    for coords, color in moves:
        if np.any(out[coords[:, 0], coords[:, 1]] != int(bg)):
            return None
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    shape = (rng.randint(GRID_MIN, GRID_MAX), rng.randint(GRID_MIN, GRID_MAX))
    bg = int(rng.choice(Color.ALL_COLORS))
    usable_colors = [color for color in Color.ALL_COLORS if int(color) != bg]

    active_count = rng.choice([2, 3, 4])
    isolated_count = rng.randint(0, max(0, 4 - active_count))
    total_frames = active_count + isolated_count
    frame_colors = rng.sample(usable_colors, total_frames)
    sprite_palette = [color for color in usable_colors if color not in frame_colors]
    sprite_colors = rng.sample(sprite_palette, total_frames)

    active_frames = _build_active_frames(shape, active_count, frame_colors[:active_count], rng)
    if active_frames is None:
        return None
    frames = active_frames[:]

    if not _add_isolated_frames(frames, shape, frame_colors[active_count:], rng):
        return None

    for index, frame in enumerate(frames):
        sprite = _place_sprite(
            frame["bbox"],
            rng.choice(SPRITE_PATTERNS),
            frame["anchor"],
            sprite_colors[index],
            rng,
        )
        if sprite is None:
            return None
        frame["sprite"] = sprite

    grid = np.full(shape, bg, dtype=int)
    for frame in sorted(frames, key=lambda frame: int(frame["z"])):
        _draw_frame(grid, frame["bbox"], frame["color"])
    for frame in frames:
        _paint_sprite(grid, frame["sprite"])

    standalone = _place_standalone_sprites(grid, frames, sprite_palette, rng)
    if standalone is None:
        return None

    for index, frame in enumerate(frames):
        owners = [
            j
            for j, other in enumerate(frames)
            if _coords_inside_bbox(frame["sprite"]["coords"], _interior_bbox(other["bbox"]))
        ]
        if owners != [index]:
            return None
    for sprite in standalone:
        if any(_coords_inside_bbox(sprite["coords"], _interior_bbox(frame["bbox"])) for frame in frames):
            return None

    expected = _expected_output(grid, frames, bg)
    if expected is None:
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        case = _generate_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError(f"failed to generate a valid case after {GEN_MAX_ATTEMPTS} attempts")
