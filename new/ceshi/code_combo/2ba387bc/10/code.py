# Combo variant of 2ba387bc (index 10)
# combo_spec: D2/3+D7/3
# Name: 颜色值排序+渐变重着色

from __future__ import annotations

"""
Design analysis for the fused variant:

- Concrete behavior:
  The original task grouped 4x4 frames and 4x4 solids, then sorted each group by
  input row and pasted the original colors into the output. Here the group split
  is unchanged, but the within-group order is now driven by a fixed visible color
  spectrum order: red -> orange -> yellow -> green -> teal -> blue. After that
  sorting, the placed objects are recolored by stack position with a fixed ramp
  (position 1/2/3/4 -> colors 5/6/9/2). Example: if frame colors appear in the
  input as yellow above red above teal, the original solver would keep that row
  order, while the fused solver outputs them as red-ranked item first, then yellow,
  then teal, and those three outputs become colors 5, 6, 9 regardless of the
  original object colors.

- Complexity calibration:
  Sorting by raw ARC ids would be invisible to a human, so the generator only uses
  colors from one fixed visible spectrum order and repeats that order across
  examples. Each group has 2-4 members with unique colors inside the group, which
  removes tie ambiguity while still leaving enough repeated evidence for humans to
  infer the ordering rule. The gradient has only four positions, matching the
  original task's object-count scale.

- Implicit assumptions audited against the original solver:
  `find_components(..., monochromatic=False)` still assumes every object is an
  isolated 4x4 connected component, so the generator keeps one-cell padding around
  every object. The 4x4 bbox assumption still holds. Frame detection by a uniform
  border plus bg-filled 2x2 center still holds. What no longer holds is the old
  assumption that output order can be derived from `(row, col)` and that pasting
  the original `sub` preserves the correct output colors, so the new solver must
  recover each object's source color, sort by visible spectrum rank, and rebuild a
  recolored frame/solid patch for output.

- Placement legality / visual cleanliness:
  The generator reserves full 4x4 footprints on a separate occupancy canvas before
  painting the visible objects, so frames cannot be overlapped through their hollow
  centers. Objects stay in-bounds, disconnected, and easy to read.

- Observability / anti-degeneracy:
  The generator intentionally assigns colors to row-sorted positions in a scrambled
  order, so row order and color order differ in every group with at least two
  objects. It also rejects any sample whose fused output would equal the original
  row-sorted output.
"""

import numpy as np
from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location

SOURCE_ORDER = (2, 7, 4, 3, 8, 1)  # red -> orange -> yellow -> green -> teal -> blue
SOURCE_RANK = {int(col): idx for idx, col in enumerate(SOURCE_ORDER)}
GRADIENT_COLORS = (5, 6, 9, 2)  # stack position 1/2/3/4


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = int(bg)
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _is_frame(sub: np.ndarray, bg: int) -> bool:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    return len(set(map(int, border.flatten()))) == 1 and bool((sub[1:3, 1:3] == int(bg)).all())


def _object_color(sub: np.ndarray, bg: int, is_frame: bool) -> int:
    if is_frame:
        return int(sub[0, 0])
    fg = sorted(set(map(int, sub[sub != int(bg)].flatten())))
    if len(fg) != 1:
        raise ValueError("solid object must have a single foreground color")
    return int(fg[0])


def _collect_objects(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    objects: list[dict[str, object]] = []
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        frame = _is_frame(sub, bg)
        color = _object_color(sub, bg, frame)
        objects.append(
            {
                "kind": "frame" if frame else "solid",
                "color": int(color),
                "r": int(r0),
                "c": int(c0),
                "sub": sub,
            }
        )
    return bg, objects


def _sort_key(obj: dict[str, object]) -> tuple[int, int, int]:
    color = int(obj["color"])
    rank = SOURCE_RANK.get(color, len(SOURCE_ORDER) + color)
    return int(rank), int(obj["r"]), int(obj["c"])


def _recolored_patch(kind: str, idx: int, bg: int) -> np.ndarray:
    col = int(GRADIENT_COLORS[min(int(idx), len(GRADIENT_COLORS) - 1)])
    return _make_frame(col, bg) if kind == "frame" else _make_solid(col)


def _baseline_main(grid: np.ndarray) -> np.ndarray:
    bg, objects = _collect_objects(grid)
    frames = [obj for obj in objects if obj["kind"] == "frame"]
    solids = [obj for obj in objects if obj["kind"] == "solid"]
    frames.sort(key=lambda obj: (int(obj["r"]), int(obj["c"])))
    solids.sort(key=lambda obj: (int(obj["r"]), int(obj["c"])))

    out = np.full((4 * max(len(frames), len(solids)), 8), bg, dtype=int)
    for i, obj in enumerate(frames):
        paste(out, np.array(obj["sub"], dtype=int), i * 4, 0)
    for i, obj in enumerate(solids):
        paste(out, np.array(obj["sub"], dtype=int), i * 4, 4)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify 4x4 objects into frames and solids, sort each group by visible source
    color order, then recolor by stack position before output.
    """
    bg, objects = _collect_objects(grid)
    frames = [obj for obj in objects if obj["kind"] == "frame"]
    solids = [obj for obj in objects if obj["kind"] == "solid"]
    frames.sort(key=_sort_key)
    solids.sort(key=_sort_key)

    out_h = 4 * max(len(frames), len(solids))
    out = np.full((out_h, 8), bg, dtype=int)

    for i, obj in enumerate(frames):
        paste(out, _recolored_patch(str(obj["kind"]), i, bg), i * 4, 0)
    for i, obj in enumerate(solids):
        paste(out, _recolored_patch(str(obj["kind"]), i, bg), i * 4, 4)

    return out


def _reserve_positions(occ: np.ndarray, n: int) -> list[tuple[int, int]]:
    patch = np.ones((4, 4), dtype=int)
    pos: list[tuple[int, int]] = []
    for _ in range(int(n)):
        r0, c0 = random_free_location(
            occ,
            patch,
            transparent=0,
            border=1,
            padding=1,
            padding_connectivity=4,
        )
        paste(occ, patch, r0, c0)
        pos.append((int(r0), int(c0)))
    return pos


def _scrambled_group_colors(n: int) -> list[int]:
    chosen = list(np.random.choice(np.array(SOURCE_ORDER, dtype=int), size=int(n), replace=False))
    chosen.sort(key=lambda col: SOURCE_RANK[int(col)])
    if n == 2:
        return [int(chosen[1]), int(chosen[0])]
    if np.random.rand() < 0.5:
        chosen = chosen[::-1]
    else:
        shift = int(np.random.randint(1, n))
        chosen = chosen[shift:] + chosen[:shift]
    return [int(col) for col in chosen]


def _paint_group(
    grid: np.ndarray,
    positions: list[tuple[int, int]],
    kind: str,
    bg: int,
) -> None:
    row_order = sorted(range(len(positions)), key=lambda idx: (positions[idx][0], positions[idx][1]))
    scrambled = _scrambled_group_colors(len(positions))
    assigned = [0] * len(positions)
    for order_idx, pos_idx in enumerate(row_order):
        assigned[pos_idx] = int(scrambled[order_idx])

    for (r0, c0), col in zip(positions, assigned):
        patch = _make_frame(col, bg) if kind == "frame" else _make_solid(col)
        paste(grid, patch, r0, c0, transparent=bg)


def _group_inversion_count(grid: np.ndarray) -> int:
    _bg, objects = _collect_objects(grid)
    total = 0
    for kind in ("frame", "solid"):
        group = [obj for obj in objects if obj["kind"] == kind]
        by_row = sorted(group, key=lambda obj: (int(obj["r"]), int(obj["c"])))
        by_color = sorted(group, key=_sort_key)
        total += sum(
            1
            for i in range(min(len(by_row), len(by_color)))
            if (int(by_row[i]["r"]), int(by_row[i]["c"])) != (int(by_color[i]["r"]), int(by_color[i]["c"]))
        )
    return int(total)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(22, 31))
    w = int(np.random.randint(22, 31))
    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=int)
    n_frames = int(np.random.randint(2, 5))
    n_solids = int(np.random.randint(2, 5))

    try:
        frame_pos = _reserve_positions(occ, n_frames)
        solid_pos = _reserve_positions(occ, n_solids)
        _paint_group(grid, frame_pos, "frame", bg)
        _paint_group(grid, solid_pos, "solid", bg)

        if _group_inversion_count(grid) == 0:
            return None
        if np.array_equal(main(grid), _baseline_main(grid)):
            return None
        return grid
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
