from __future__ import annotations

import random
from collections import Counter

import numpy as np


ALL_COLORS = tuple(range(10))
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400

GAP_INFO = {
    "top": {"gap": (0, 1), "stamp": (2, 1)},
    "bottom": {"gap": (2, 1), "stamp": (0, 1)},
    "left": {"gap": (1, 0), "stamp": (1, 2)},
    "right": {"gap": (1, 2), "stamp": (1, 0)},
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _edge_side(r: int, c: int, h: int, w: int) -> str | None:
    if r == 0:
        return "top"
    if r == h - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == w - 1:
        return "right"
    return None


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _boxes_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _mark_halo(blocked: np.ndarray, box: tuple[int, int, int, int]) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = box
    blocked[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True


def _paint_object(
    out: np.ndarray, color: int, top: int, left: int, gap_side: str, bg: int
) -> None:
    out[top : top + 3, left : left + 3] = int(color)
    gr, gc = GAP_INFO[gap_side]["gap"]
    out[top + gr, left + gc] = int(bg)


def _dest_box_for_marker(side: str, pos: int, h: int, w: int) -> tuple[int, int, int, int]:
    if side == "top":
        return 0, pos - 1, 2, pos + 1
    if side == "bottom":
        return h - 3, pos - 1, h - 1, pos + 1
    if side == "left":
        return pos - 1, 0, pos + 1, 2
    return pos - 1, w - 3, pos + 1, w - 1


def _start_box_from_spec(spec: dict[str, int]) -> tuple[int, int, int, int]:
    return spec["r0"], spec["c0"], spec["r0"] + 2, spec["c0"] + 2


def _corridor_box(
    spec: dict[str, int], h: int, w: int
) -> tuple[int, int, int, int]:
    r0, c0 = spec["r0"], spec["c0"]
    side = spec["gap_side"]
    if side == "top":
        return 0, c0, r0 + 2, c0 + 2
    if side == "bottom":
        return r0, c0, h - 1, c0 + 2
    if side == "left":
        return r0, 0, r0 + 2, c0 + 2
    return r0, c0, r0 + 2, w - 1


def _target_top_left(
    spec: dict[str, int], marker: dict[str, int], h: int, w: int
) -> tuple[int, int] | None:
    side = spec["gap_side"]
    r0, c0 = spec["r0"], spec["c0"]
    if marker["side"] != side:
        return None

    if side in ("top", "bottom") and c0 + 1 != marker["pos"]:
        return None
    if side in ("left", "right") and r0 + 1 != marker["pos"]:
        return None

    if side == "top":
        return 0, c0
    if side == "bottom":
        return h - 3, c0
    if side == "left":
        return r0, 0
    return r0, w - 3


def _extract_problem(
    grid: np.ndarray,
) -> tuple[int, dict[int, dict[str, int]], list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    markers: dict[int, dict[str, int]] = {}
    for r in range(h):
        for c in range(w):
            side = _edge_side(r, c, h, w)
            if side is None:
                continue
            color = int(g[r, c])
            if color == bg:
                continue
            if color in markers:
                raise ValueError("marker colors must be unique on the boundary")
            pos = c if side in ("top", "bottom") else r
            markers[color] = {"side": side, "pos": int(pos), "r": int(r), "c": int(c)}

    objects: list[dict[str, int]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        coords = [
            (int(r), int(c))
            for r, c in np.argwhere(g == int(color))
            if 0 < int(r) < h - 1 and 0 < int(c) < w - 1
        ]
        if not coords:
            continue
        if len(coords) != 8:
            raise ValueError("each object color must occupy exactly eight interior cells")
        r0, c0, r1, c1 = _bbox(coords)
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            raise ValueError("each object must have a 3x3 bbox")
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if int(np.count_nonzero(patch == int(color))) != 8:
            raise ValueError("object patch must contain exactly eight object-colored cells")
        if bool(((patch != int(bg)) & (patch != int(color))).any()):
            raise ValueError("object patch must be monochrome with one bg hole")

        gap_side = None
        for side, info in GAP_INFO.items():
            expected = np.full((3, 3), int(color), dtype=int)
            gr, gc = info["gap"]
            expected[gr, gc] = int(bg)
            if np.array_equal(patch, expected):
                gap_side = side
                break
        if gap_side is None:
            raise ValueError("object hole must be on one of the four side centers")

        objects.append(
            {
                "color": int(color),
                "r0": int(r0),
                "c0": int(c0),
                "gap_side": gap_side,
            }
        )

    return bg, markers, objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each holed square toward its matching edge marker, then leave a mirrored single-pixel stamp behind.

    Input structure:
    - The background is the most common color.
    - The grid boundary contains isolated single-pixel color markers, with each marker color unique.
    - Inside the grid are monochrome 3x3 square objects with exactly one background hole at the top-middle, bottom-middle, left-middle, or right-middle cell.
    - Every object color matches exactly one boundary marker color.

    Rule:
    - The hole side encodes a motion direction: top hole -> move up, bottom hole -> move down, left hole -> move left, right hole -> move right.
    - If the same-colored boundary marker lies on that destination edge and is aligned with the object's center line, slide the object straight to that edge without rotating it.
    - When an object slides, clear its old 3x3 footprint to background except for a single same-colored stamp at the cell opposite the hole.
    - If the matching marker is on a different edge, keep the object in place; all boundary markers stay unchanged.

    Key insight:
    - The hole gives the direction, the matching boundary marker chooses whether the move is valid, and the old location keeps only the hole's opposite-side mirror cell.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, markers, objects = _extract_problem(g)

    out = np.full_like(g, int(bg))
    out[0, :] = g[0, :]
    out[h - 1, :] = g[h - 1, :]
    out[:, 0] = g[:, 0]
    out[:, w - 1] = g[:, w - 1]

    for spec in objects:
        color = int(spec["color"])
        marker = markers.get(color)
        target = None if marker is None else _target_top_left(spec, marker, h, w)
        if target is None:
            _paint_object(out, color, spec["r0"], spec["c0"], spec["gap_side"], bg)
            continue

        tr, tc = target
        _paint_object(out, color, tr, tc, spec["gap_side"], bg)
        sr, sc = GAP_INFO[spec["gap_side"]]["stamp"]
        out[spec["r0"] + sr, spec["c0"] + sc] = int(color)

    return out


def _balanced_sides(rng: random.Random, count: int) -> list[str]:
    pool = list(SIDES)
    while len(pool) < count:
        pool.extend(SIDES)
    rng.shuffle(pool)
    return pool[:count]


def _balanced_flags(rng: random.Random, count: int) -> list[bool]:
    flags = [True] * (count // 2) + [False] * (count // 2)
    if count % 2 == 1:
        flags.append(bool(rng.randint(0, 1)))
    rng.shuffle(flags)
    if all(flags):
        flags[-1] = False
        rng.shuffle(flags)
    if not any(flags):
        flags[-1] = True
        rng.shuffle(flags)
    return flags


def _marker_position_candidates(side: str, h: int, w: int) -> list[int]:
    if side in ("top", "bottom"):
        return list(range(1, w - 1))
    return list(range(1, h - 1))


def _choose_marker_layout(
    rng: random.Random,
    h: int,
    w: int,
    specs: list[dict[str, int | str | bool]],
) -> bool:
    marker_cells: set[tuple[int, int]] = set()
    edge_positions: dict[str, list[int]] = {side: [] for side in SIDES}
    reserved_dest_boxes: list[tuple[int, int, int, int]] = []

    order = sorted(range(len(specs)), key=lambda idx: (not bool(specs[idx]["movable"]), idx))
    for idx in order:
        spec = specs[idx]
        side = str(spec["marker_side"])
        candidates = _marker_position_candidates(side, h, w)
        rng.shuffle(candidates)
        placed = False
        for pos in candidates:
            if any(abs(int(pos) - prev) < 3 for prev in edge_positions[side]):
                continue
            if side == "top":
                cell = (0, pos)
            elif side == "bottom":
                cell = (h - 1, pos)
            elif side == "left":
                cell = (pos, 0)
            else:
                cell = (pos, w - 1)

            if cell in marker_cells:
                continue

            dest_box = None
            if bool(spec["movable"]):
                dest_box = _dest_box_for_marker(side, pos, h, w)
                if any(_boxes_overlap(dest_box, box) for box in reserved_dest_boxes):
                    continue
                if any(
                    dest_box[0] <= mr <= dest_box[2] and dest_box[1] <= mc <= dest_box[3]
                    for mr, mc in marker_cells
                ):
                    continue
            else:
                conflict = False
                for box in reserved_dest_boxes:
                    if box[0] <= cell[0] <= box[2] and box[1] <= cell[1] <= box[3]:
                        conflict = True
                        break
                if conflict:
                    continue

            spec["marker_pos"] = int(pos)
            spec["marker_r"] = int(cell[0])
            spec["marker_c"] = int(cell[1])
            marker_cells.add(cell)
            edge_positions[side].append(int(pos))
            if dest_box is not None:
                reserved_dest_boxes.append(dest_box)
            placed = True
            break

        if not placed:
            return False

    return True


def _choose_start_positions(
    rng: random.Random,
    h: int,
    w: int,
    specs: list[dict[str, int | str | bool]],
) -> bool:
    blocked = np.zeros((h, w), dtype=bool)

    for spec in specs:
        if not bool(spec["movable"]):
            continue
        _mark_halo(blocked, _dest_box_for_marker(str(spec["marker_side"]), int(spec["marker_pos"]), h, w))

    movable_indices = [idx for idx, spec in enumerate(specs) if bool(spec["movable"])]
    rng.shuffle(movable_indices)
    for idx in movable_indices:
        spec = specs[idx]
        side = str(spec["gap_side"])
        pos = int(spec["marker_pos"])

        candidates: list[tuple[int, int]] = []
        if side == "top":
            c0 = pos - 1
            candidates = [(r0, c0) for r0 in range(3, h - 4)]
        elif side == "bottom":
            c0 = pos - 1
            candidates = [(r0, c0) for r0 in range(2, h - 5)]
        elif side == "left":
            r0 = pos - 1
            candidates = [(r0, c0) for c0 in range(3, w - 4)]
        else:
            r0 = pos - 1
            candidates = [(r0, c0) for c0 in range(2, w - 5)]

        rng.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            spec["r0"] = int(r0)
            spec["c0"] = int(c0)
            start_box = _start_box_from_spec(spec)  # type: ignore[arg-type]
            if bool(blocked[start_box[0] : start_box[2] + 1, start_box[1] : start_box[3] + 1].any()):
                continue
            corridor = _corridor_box(spec, h, w)  # type: ignore[arg-type]
            _mark_halo(blocked, corridor)
            placed = True
            break
        if not placed:
            return False

    static_indices = [idx for idx, spec in enumerate(specs) if not bool(spec["movable"])]
    rng.shuffle(static_indices)
    all_positions = [(r0, c0) for r0 in range(2, h - 4) for c0 in range(2, w - 4)]

    for idx in static_indices:
        spec = specs[idx]
        rng.shuffle(all_positions)
        placed = False
        for r0, c0 in all_positions:
            spec["r0"] = int(r0)
            spec["c0"] = int(c0)
            start_box = _start_box_from_spec(spec)  # type: ignore[arg-type]
            if bool(blocked[start_box[0] : start_box[2] + 1, start_box[1] : start_box[3] + 1].any()):
                continue
            _mark_halo(blocked, start_box)
            placed = True
            break
        if not placed:
            return False

    return True


def _build_grid(
    h: int, w: int, bg: int, specs: list[dict[str, int | str | bool]]
) -> np.ndarray:
    grid = np.full((h, w), int(bg), dtype=int)
    for spec in specs:
        color = int(spec["color"])
        grid[int(spec["marker_r"]), int(spec["marker_c"])] = int(color)
    for spec in specs:
        _paint_object(
            grid,
            int(spec["color"]),
            int(spec["r0"]),
            int(spec["c0"]),
            str(spec["gap_side"]),
            int(bg),
        )
    return grid


def _make_specs(rng: random.Random, count: int, colors: list[int]) -> list[dict[str, int | str | bool]]:
    gap_sides = _balanced_sides(rng, count)
    movable_flags = _balanced_flags(rng, count)
    specs: list[dict[str, int | str | bool]] = []

    for color, gap_side, movable in zip(colors, gap_sides, movable_flags):
        if movable:
            marker_side = gap_side
        else:
            other_sides = [side for side in SIDES if side != gap_side]
            marker_side = rng.choice(other_sides)
        specs.append(
            {
                "color": int(color),
                "gap_side": str(gap_side),
                "marker_side": str(marker_side),
                "movable": bool(movable),
            }
        )
    rng.shuffle(specs)
    return specs


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    obj_count = rng.randint(3, 5)
    side_low = 14 + max(0, obj_count - 3) * 2
    h = rng.randint(side_low, min(30, side_low + 6))
    w = rng.randint(side_low, min(30, side_low + 6))

    bg = int(rng.choice(ALL_COLORS))
    colors = [color for color in ALL_COLORS if color != bg]
    rng.shuffle(colors)
    colors = colors[:obj_count]

    specs = _make_specs(rng, obj_count, colors)
    if not _choose_marker_layout(rng, h, w, specs):
        return None
    if not _choose_start_positions(rng, h, w, specs):
        return None

    grid = _build_grid(h, w, bg, specs)

    if _get_bg(grid) != bg:
        return None

    try:
        _, markers, objects = _extract_problem(grid)
        out = main(grid)
    except Exception:
        return None

    if len(markers) != obj_count or len(objects) != obj_count:
        return None

    by_color = {int(obj["color"]): obj for obj in objects}
    for spec in specs:
        color = int(spec["color"])
        obj = by_color.get(color)
        marker = markers.get(color)
        if obj is None or marker is None:
            return None
        if str(obj["gap_side"]) != str(spec["gap_side"]):
            return None
        if str(marker["side"]) != str(spec["marker_side"]):
            return None
        if int(obj["r0"]) != int(spec["r0"]) or int(obj["c0"]) != int(spec["c0"]):
            return None

    if out.shape != grid.shape:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated retries")
