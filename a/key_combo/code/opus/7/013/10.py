from __future__ import annotations

from importlib import util as importlib_util
from pathlib import Path
import random

import numpy as np


def _load_arc_utils():
    here = Path(__file__).resolve()
    candidates = [
        here.parents[6] / "new" / "solvers" / "arc_utils.py",
        Path("/home/lixinyu/snap/new/solvers/arc_utils.py"),
    ]
    for path in candidates:
        if path.exists():
            spec = importlib_util.spec_from_file_location("_shared_arc_utils_10", path)
            module = importlib_util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("arc_utils.py not found")


_ARC = _load_arc_utils()
Color = _ARC.Color
find_components = _ARC.find_components
get_bg = _ARC.get_bg
paste = _ARC.paste
random_free_location = _ARC.random_free_location


BG = Color.BLACK
FRAME = Color.GREY
MOTIF = Color.RED
GEN_RETRIES = 400
PLACEMENT_PADDING = 2


def _frame_axis(r0: int, c0: int, r1: int, c1: int) -> tuple[str, int]:
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    assert inner_h >= 3 and inner_w >= 3
    assert inner_h != inner_w
    if inner_w > inner_h:
        return "vertical", c0 + 1 + inner_w // 2
    return "horizontal", r0 + 1 + inner_h // 2


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    coords += [(r, c1) for r in range(r0 + 1, r1 + 1)]
    coords += [(r1, c) for c in range(c1 - 1, c0 - 1, -1)]
    coords += [(r, c0) for r in range(r1 - 1, r0, -1)]
    return coords


def _source_side(axis_kind: str, axis: int, gap: tuple[int, int]) -> str:
    gr, gc = gap
    if axis_kind == "vertical":
        if gc < axis:
            return "left"
        if gc > axis:
            return "right"
        raise AssertionError("door gap may not lie on the mirror axis")
    if gr < axis:
        return "top"
    if gr > axis:
        return "bottom"
    raise AssertionError("door gap may not lie on the mirror axis")


def _side_filter(coords: np.ndarray, axis_kind: str, axis: int, side: str) -> np.ndarray:
    if coords.size == 0:
        return coords
    if axis_kind == "vertical":
        if side == "left":
            return coords[coords[:, 1] < axis]
        return coords[coords[:, 1] > axis]
    if side == "top":
        return coords[coords[:, 0] < axis]
    return coords[coords[:, 0] > axis]


def _reflect(coords: np.ndarray, axis_kind: str, axis: int) -> np.ndarray:
    if coords.size == 0:
        return coords.copy()
    reflected = coords.copy()
    if axis_kind == "vertical":
        reflected[:, 1] = 2 * int(axis) - reflected[:, 1]
    else:
        reflected[:, 0] = 2 * int(axis) - reflected[:, 0]
    return reflected


def _frame_records(grid: np.ndarray) -> list[dict]:
    frames = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[FRAME],
    )
    records: list[dict] = []
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        axis_kind, axis = _frame_axis(r0, c0, r1, c1)
        perimeter = _perimeter_coords(r0, c0, r1, c1)
        gap_cells = [(r, c) for r, c in perimeter if int(grid[r, c]) == BG]
        assert len(gap_cells) == 1, "each frame must have exactly one perimeter gap"
        gap = gap_cells[0]
        for r, c in perimeter:
            if (r, c) != gap:
                assert int(grid[r, c]) == FRAME, "frame perimeter must be grey except at the gap"

        red = np.argwhere(grid[r0 + 1 : r1, c0 + 1 : c1] == MOTIF)
        red[:, 0] += r0 + 1
        red[:, 1] += c0 + 1
        side = _source_side(axis_kind, axis, gap)
        source_red = _side_filter(red, axis_kind, axis, side)
        source_set = {tuple(map(int, rc)) for rc in source_red}
        other_red = np.array([coord for coord in red if tuple(map(int, coord)) not in source_set], dtype=int)
        records.append(
            {
                "bbox": (r0, c0, r1, c1),
                "axis_kind": axis_kind,
                "axis": axis,
                "gap": gap,
                "side": side,
                "red": red,
                "source_red": source_red,
                "other_red": other_red.reshape((-1, 2)) if other_red.size else np.zeros((0, 2), dtype=int),
            }
        )
    assert records, "expected at least one frame"
    return records


def _paint_frame_variant(
    grid: np.ndarray,
    records: list[dict],
    *,
    mirror_all_red: bool = False,
    swap_source_side: bool = False,
    swap_axis_kind: bool = False,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for rec in records:
        r0, c0, r1, c1 = rec["bbox"]
        axis_kind = rec["axis_kind"]
        axis = rec["axis"]
        gap = rec["gap"]
        side = rec["side"]
        red = rec["red"]
        if swap_axis_kind:
            if axis_kind == "vertical":
                axis_kind = "horizontal"
                axis = r0 + 1 + (r1 - r0 - 1) // 2
                side = "top" if gap[0] < axis else "bottom"
            else:
                axis_kind = "vertical"
                axis = c0 + 1 + (c1 - c0 - 1) // 2
                side = "left" if gap[1] < axis else "right"
        elif swap_source_side:
            if side == "left":
                side = "right"
            elif side == "right":
                side = "left"
            elif side == "top":
                side = "bottom"
            else:
                side = "top"

        chosen = red if mirror_all_red else _side_filter(red, axis_kind, axis, side)
        reflected = _reflect(chosen, axis_kind, axis)
        for rr, cc in reflected:
            if r0 < int(rr) < r1 and c0 < int(cc) < c1:
                out[int(rr), int(cc)] = MOTIF
        out[gap] = FRAME
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each gapped frame by mirroring only the door-indicated half across the aspect-ratio axis.

    Input structure:
    - The background is black.
    - The foreground contains several one-pixel-thick grey rectangular frames, each with exactly one one-pixel gap on its perimeter.
    - Inside each frame are red motif pixels; only the red pixels on the half indicated by the gap are meant to be mirrored, while any red pixels already on the other half stay unchanged.

    Rule:
    - For each grey frame, compute the interior width and height.
    - If the interior is wider than tall, use the interior vertical midline as the mirror axis; if it is taller than wide, use the interior horizontal midline.
    - The gap marks the source half: left or right of a vertical axis, top or bottom of a horizontal axis.
    - Mirror the red pixels from that source half across the chosen axis into the opposite half, keep all existing red pixels, and close the gap by repainting it grey.

    Key insight:
    - The frame aspect ratio chooses which axis to use, but the door gap chooses which half supplies the mirrored motif.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BG
    return _paint_frame_variant(g, _frame_records(g))


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _random_connected_shape(
    allowed: set[tuple[int, int]],
    size: int,
    *,
    require_spread: bool,
) -> set[tuple[int, int]] | None:
    if len(allowed) < size:
        return None
    cells = list(allowed)
    random.shuffle(cells)
    for seed in cells:
        shape = {seed}
        while len(shape) < size:
            frontier = []
            for cell in shape:
                for nb in _neighbors4(cell):
                    if nb in allowed and nb not in shape:
                        frontier.append(nb)
            if not frontier:
                break
            shape.add(random.choice(frontier))
        if len(shape) != size:
            continue
        rows = {r for r, _ in shape}
        cols = {c for _, c in shape}
        if require_spread and (len(rows) < 2 or len(cols) < 2):
            continue
        return shape
    return None


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    target = {tuple(map(int, rc)) for rc in coords}
    seen = {start}
    stack = [start]
    while stack:
        cell = stack.pop()
        for nb in _neighbors4(cell):
            if nb in target and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(target)


def _side_cells(inner_h: int, inner_w: int, axis_kind: str, side: str) -> set[tuple[int, int]]:
    if axis_kind == "vertical":
        axis = inner_w // 2
        cols = range(0, axis) if side == "left" else range(axis + 1, inner_w)
        return {(r, c) for r in range(inner_h) for c in cols}
    axis = inner_h // 2
    rows = range(0, axis) if side == "top" else range(axis + 1, inner_h)
    return {(r, c) for r in rows for c in range(inner_w)}


def _reflect_rel(coords: set[tuple[int, int]], axis_kind: str, inner_h: int, inner_w: int) -> set[tuple[int, int]]:
    if axis_kind == "vertical":
        axis = inner_w // 2
        return {(r, 2 * axis - c) for r, c in coords}
    axis = inner_h // 2
    return {(2 * axis - r, c) for r, c in coords}


def _door_position(inner_h: int, inner_w: int, axis_kind: str, side: str) -> tuple[int, int]:
    if axis_kind == "vertical":
        axis = inner_w // 2
        if side == "left":
            cols = list(range(1, axis + 1))
        else:
            cols = list(range(axis + 2, inner_w + 1))
        return (0 if random.random() < 0.5 else inner_h + 1, random.choice(cols))
    axis = inner_h // 2
    if side == "top":
        rows = list(range(1, axis + 1))
    else:
        rows = list(range(axis + 2, inner_h + 1))
    return (random.choice(rows), 0 if random.random() < 0.5 else inner_w + 1)


def _build_frame_patch(
    axis_kind: str,
    inner_h: int,
    inner_w: int,
    source_side: str,
    *,
    force_distractor: bool,
) -> np.ndarray | None:
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = FRAME
    patch[-1, :] = FRAME
    patch[:, 0] = FRAME
    patch[:, -1] = FRAME

    source_cells = _side_cells(inner_h, inner_w, axis_kind, source_side)
    source_size = random.randint(3, 6)
    source_shape = _random_connected_shape(source_cells, source_size, require_spread=True)
    if source_shape is None:
        return None

    mirrored_shape = _reflect_rel(source_shape, axis_kind, inner_h, inner_w)
    if mirrored_shape & source_shape:
        return None

    other_side = {
        "left": "right",
        "right": "left",
        "top": "bottom",
        "bottom": "top",
    }[source_side]
    other_cells = _side_cells(inner_h, inner_w, axis_kind, other_side) - mirrored_shape
    distractor_shape: set[tuple[int, int]] = set()
    if force_distractor or random.random() < 0.7:
        min_size = 2 if len(other_cells) >= 2 else 0
        max_size = min(4, len(other_cells))
        if min_size <= max_size and max_size > 0:
            distractor_size = random.randint(min_size, max_size)
            distractor_shape = _random_connected_shape(other_cells, distractor_size, require_spread=False) or set()
    if force_distractor and not distractor_shape:
        return None

    if mirrored_shape & distractor_shape:
        return None

    for r, c in source_shape | distractor_shape:
        patch[r + 1, c + 1] = MOTIF

    gap_r, gap_c = _door_position(inner_h, inner_w, axis_kind, source_side)
    patch[gap_r, gap_c] = BG
    return patch


def _sample_frame_specs(n_frames: int) -> list[tuple[str, int, int, str]]:
    specs: list[tuple[str, int, int, str]] = []
    orientations = ["wide", "tall"]
    if n_frames == 3:
        orientations.append(random.choice(["wide", "tall"]))
    random.shuffle(orientations)
    side_labels = []
    for orient in orientations:
        if orient == "wide":
            inner_h = random.choice([3, 5])
            feasible_w = [w for w in [5, 7, 9] if w > inner_h and w >= int(np.ceil(1.5 * inner_h))]
            inner_w = random.choice(feasible_w)
            side = random.choice(["left", "right"])
            specs.append(("vertical", inner_h, inner_w, side))
        else:
            inner_w = random.choice([3, 5])
            feasible_h = [h for h in [5, 7, 9] if h > inner_w and h >= int(np.ceil(1.5 * inner_w))]
            inner_h = random.choice(feasible_h)
            side = random.choice(["top", "bottom"])
            specs.append(("horizontal", inner_h, inner_w, side))
        side_labels.append(side)
    if len(set(side_labels)) == 1:
        axis_kind, inner_h, inner_w, side = specs[-1]
        if side in ("left", "right"):
            side = "left" if side == "right" else "right"
        else:
            side = "top" if side == "bottom" else "bottom"
        specs[-1] = (axis_kind, inner_h, inner_w, side)
    return specs


def _attempt_generate() -> np.ndarray | None:
    n_frames = random.randint(2, 3)
    specs = _sample_frame_specs(n_frames)
    patches: list[np.ndarray] = []
    for idx, (axis_kind, inner_h, inner_w, side) in enumerate(specs):
        patch = _build_frame_patch(
            axis_kind,
            inner_h,
            inner_w,
            side,
            force_distractor=(idx == 0),
        )
        if patch is None:
            return None
        patches.append(patch)

    height = random.randint(22, 30)
    width = random.randint(22, 30)
    grid = np.zeros((height, width), dtype=int)
    occupancy = np.zeros_like(grid)

    order = list(range(len(patches)))
    random.shuffle(order)
    order.sort(key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    placements: dict[int, tuple[int, int]] = {}
    for idx in order:
        block = np.ones(patches[idx].shape, dtype=int)
        try:
            r0, c0 = random_free_location(
                occupancy,
                block,
                transparent=BG,
                padding=PLACEMENT_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        placements[idx] = (r0, c0)
        paste(occupancy, block, r0, c0, transparent=BG)

    for idx, patch in enumerate(patches):
        r0, c0 = placements[idx]
        paste(grid, patch, r0, c0, transparent=BG)

    _validate_instance(grid)
    return grid


def _validate_instance(grid: np.ndarray) -> None:
    records = _frame_records(grid)
    assert 2 <= len(records) <= 3
    assert {rec["axis_kind"] for rec in records} == {"vertical", "horizontal"}
    assert len({rec["side"] for rec in records}) >= 2

    has_other_half_red = False
    for rec in records:
        r0, c0, r1, c1 = rec["bbox"]
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        assert inner.shape[0] >= 3 and inner.shape[1] >= 3
        assert rec["source_red"].shape[0] >= 3
        source_mask = np.zeros_like(grid, dtype=bool)
        for rr, cc in rec["source_red"]:
            source_mask[int(rr), int(cc)] = True
        assert _connected(source_mask)

        reflected = _reflect(rec["source_red"], rec["axis_kind"], rec["axis"])
        targets = {tuple(map(int, rc)) for rc in reflected}
        assert not targets & {tuple(map(int, rc)) for rc in rec["source_red"]}
        for rr, cc in targets:
            assert r0 < rr < r1 and c0 < cc < c1
        other = {tuple(map(int, rc)) for rc in rec["other_red"]}
        if other:
            has_other_half_red = True
            assert not (targets & other), "distractors may not occupy mirrored target cells"

    assert has_other_half_red, "at least one frame must need the door-side cue"

    expected = main(grid)
    naive = _paint_frame_variant(grid, records, mirror_all_red=True)
    wrong_side = _paint_frame_variant(grid, records, swap_source_side=True)
    wrong_axis = _paint_frame_variant(grid, records, swap_axis_kind=True)
    assert not np.array_equal(expected, naive), "door cue must affect the result"
    assert not np.array_equal(expected, wrong_side), "source-side choice must be necessary"
    assert not np.array_equal(expected, wrong_axis), "aspect-ratio axis choice must be necessary"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
