# Combo variant of 2ba387bc (index 11)
# combo_spec: D3/0+D11/0
# Name: 可变尺寸+非标准尺寸干扰

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SOLVERS_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(_SOLVERS_DIR) not in sys.path:
    sys.path.append(str(_SOLVERS_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace:
#    - Original task: every relevant object is an isolated 4x4 square; multi-color
#      connected-component detection is safe, and output rows are fixed 4 cells high.
#    - D3/0 changes the valid objects to square sizes in {3,4,5}. Example:
#      a 3x3 frame above a 5x5 solid still occupy the first output "row", but that row
#      must now be 5 cells high and the output width must be 2 * max_size.
#    - D11/0 adds extra colored regions whose shapes are not valid 3..5 squares, such
#      as 2x3 or 1x5 bars. At least one distractor is attached orthogonally to a valid
#      square using a different color. A multi-color component solver would merge them
#      and lose the square. Therefore the combo solver must detect monochromatic
#      components first, then filter by the valid size range and frame/solid pattern.
# b) Complexity calibration:
#    - Valid sizes are restricted to 3, 4, 5 only, so the rule stays inferable.
#    - Distractors use a small rectangle vocabulary and unique colors, so they are
#      visually obvious but not ambiguous.
#    - The generator forces at least one non-4 valid square, at least two distinct
#      valid sizes overall, and at least one attached distractor. This prevents the
#      sample set from collapsing back to the source task or to the D3-only variant.
# c) Hidden-assumption audit of the original main():
#    - `monochromatic=False` assumed touching different colors still belonged to one
#      meaningful object; attached distractors break that assumption.
#    - `bbox_h(b) == bbox_w(b) == 4` assumed a single fixed size; D3 breaks it.
#    - `sub[1:3, 1:3] == bg` assumed a fixed 2x2 interior; variable sizes require
#      `sub[1:-1, 1:-1] == bg`.
#    - `outH = 4 * ...`, `outW = 8`, and row starts at `i * 4` all fail once objects
#      can be 3x3..5x5.
#    - Treating every surviving non-frame as a solid is too weak with distractors
#      present, so the combo solver makes the solid test explicit.
# d) Placement legality:
#    - Valid squares are placed with a fully reserved square footprint so frame holes
#      remain empty and no later object can land inside them.
#    - Distractors never overlap colored cells. Attached distractors are accepted only
#      if they touch exactly one chosen valid square and no other object.
# e) Visual observability:
#    - Humans can see both changed dimensions directly: valid objects appear in mixed
#      square sizes, and extra non-square bars/blocks are ignored by the output.
#    - Attached distractors make it observable that "connected blob" is not the right
#      unit; the solver must reason at the level of valid monochrome squares.
# f) Visual cleanliness:
#    - Colors are unique per object, valid squares keep at least one-cell spacing from
#      one another, and distractors are simple rectangles, so the scene stays readable.
# g) New solver assumptions and generator guarantees:
#    - Every relevant object is a monochromatic square with side length 3..5.
#    - Frames have a uniform 1-pixel border and a fully background interior.
#    - Solids are uniformly filled with one non-background color.
#    - The generator guarantees all of the above and rejects any sample where the combo
#      output accidentally matches a D3-only or size-only baseline.


VALID_MIN_SIZE = 3
VALID_MAX_SIZE = 5
DISTRACTOR_SHAPES: tuple[tuple[int, int], ...] = (
    (1, 4),
    (4, 1),
    (1, 5),
    (5, 1),
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (2, 5),
    (5, 2),
)


def _make_frame(size: int, col: int, bg: int = 0) -> np.ndarray:
    a = np.full((int(size), int(size)), int(col), dtype=int)
    a[1:-1, 1:-1] = int(bg)
    return a


def _make_solid(size: int, col: int) -> np.ndarray:
    return np.full((int(size), int(size)), int(col), dtype=int)


def _classify_square(sub: np.ndarray, bg: int) -> str | None:
    h, w = map(int, sub.shape)
    if h != w or h < VALID_MIN_SIZE or h > VALID_MAX_SIZE:
        return None

    vals = set(map(int, sub.flatten().tolist()))
    if len(vals) == 1 and int(next(iter(vals))) != int(bg):
        return "solid"

    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    border_cols = set(map(int, border.tolist()))
    if len(border_cols) != 1:
        return None
    border_col = int(next(iter(border_cols)))
    if border_col == int(bg):
        return None
    if bool((sub[1:-1, 1:-1] == int(bg)).all()):
        return "frame"
    return None


def _collect_valid_objects(
    grid: np.ndarray,
    bg: int,
    *,
    monochromatic: bool,
    min_size: int = VALID_MIN_SIZE,
    max_size: int = VALID_MAX_SIZE,
) -> list[dict]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=monochromatic, return_format="dict")
    objects: list[dict] = []

    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        if h != w or h < int(min_size) or h > int(max_size):
            continue

        r0, c0, _r1, _c1 = b
        sub = crop_bbox(grid, b)
        kind = _classify_square(sub, bg)
        if kind is None:
            continue

        objects.append(
            {
                "kind": kind,
                "size": h,
                "r0": int(r0),
                "c0": int(c0),
                "sprite": sub,
            }
        )

    return objects


def _assemble_output(objects: list[dict], bg: int) -> np.ndarray:
    frames = [obj for obj in objects if obj["kind"] == "frame"]
    solids = [obj for obj in objects if obj["kind"] == "solid"]
    frames.sort(key=lambda obj: (int(obj["r0"]), int(obj["c0"])))
    solids.sort(key=lambda obj: (int(obj["r0"]), int(obj["c0"])))

    if not frames and not solids:
        return np.zeros((0, 0), dtype=int)

    max_size = max(int(obj["size"]) for obj in objects)
    row_count = max(len(frames), len(solids))
    row_heights = [
        max(
            int(frames[i]["size"]) if i < len(frames) else 0,
            int(solids[i]["size"]) if i < len(solids) else 0,
        )
        for i in range(row_count)
    ]

    out = np.full((int(sum(row_heights)), int(2 * max_size)), int(bg), dtype=int)
    rr = 0
    for i, band_h in enumerate(row_heights):
        if i < len(frames):
            paste(out, frames[i]["sprite"], int(rr), 0)
        if i < len(solids):
            paste(out, solids[i]["sprite"], int(rr), int(max_size))
        rr += int(band_h)
    return out


def _same_output(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _size_only_baseline(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    objs = _collect_valid_objects(g, bg, monochromatic=True, min_size=4, max_size=4)
    return _assemble_output(objs, bg)


def _connectivity_only_baseline(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    objs = _collect_valid_objects(g, bg, monochromatic=False, min_size=VALID_MIN_SIZE, max_size=VALID_MAX_SIZE)
    return _assemble_output(objs, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort only the valid 3x3..5x5 squares into two columns by type.

    Combo rule:
    - Relevant objects are monochromatic squares of size 3..5 only.
    - Frames have a 1-pixel border and a background interior.
    - Solids are fully filled with one non-background color.
    - Extra colored regions with any other shape/size are distractors and ignored.
    - Frames are stacked in the left output column, solids in the right.
    - Column width is the maximum valid square size in the input.
    - Each output row-band height is the larger size of the paired frame/solid item.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    objects = _collect_valid_objects(g, bg, monochromatic=True)
    return _assemble_output(objects, bg)


def _sample_sizes(total: int) -> list[int]:
    while True:
        sizes = [int(np.random.randint(VALID_MIN_SIZE, VALID_MAX_SIZE + 1)) for _ in range(int(total))]
        if any(size != 4 for size in sizes) and len(set(sizes)) >= 2:
            return sizes


def _bbox_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(int(r0), int(r1) + 1) for c in range(int(c0), int(c1) + 1)}


def _can_place_attached(
    g: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    target_bbox: tuple[int, int, int, int],
    bg: int,
) -> bool:
    h, w = map(int, patch.shape)
    H, W = g.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    if bool((g[r0 : r0 + h, c0 : c0 + w] != int(bg)).any()):
        return False

    target_cells = _bbox_cells(target_bbox)
    touched_target = False
    touched_other = False
    for rr in range(int(r0), int(r0 + h)):
        for cc in range(int(c0), int(c0 + w)):
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if int(g[nr, nc]) == int(bg):
                    continue
                if (nr, nc) in target_cells:
                    touched_target = True
                else:
                    touched_other = True
    return touched_target and not touched_other


def _attached_location(
    g: np.ndarray,
    patch: np.ndarray,
    target_bbox: tuple[int, int, int, int],
    bg: int,
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = target_bbox
    h, w = map(int, patch.shape)
    candidates: list[tuple[int, int]] = []

    if c0 - w >= 0:
        for rr in range(int(r0 - h + 1), int(r1) + 1):
            candidates.append((rr, int(c0 - w)))
    if c1 + 1 + w <= g.shape[1]:
        for rr in range(int(r0 - h + 1), int(r1) + 1):
            candidates.append((rr, int(c1 + 1)))
    if r0 - h >= 0:
        for cc in range(int(c0 - w + 1), int(c1) + 1):
            candidates.append((int(r0 - h), cc))
    if r1 + 1 + h <= g.shape[0]:
        for cc in range(int(c0 - w + 1), int(c1) + 1):
            candidates.append((int(r1 + 1), cc))

    np.random.shuffle(candidates)
    for rr, cc in candidates:
        if _can_place_attached(g, patch, int(rr), int(cc), target_bbox, bg):
            return int(rr), int(cc)
    return None


def _place_distractor(
    g: np.ndarray,
    patch: np.ndarray,
    bg: int,
    anchors: list[tuple[int, int, int, int]],
    *,
    force_attach: bool,
) -> bool:
    anchor_order = list(range(len(anchors)))
    np.random.shuffle(anchor_order)

    if force_attach:
        for idx in anchor_order:
            loc = _attached_location(g, patch, anchors[idx], bg)
            if loc is not None:
                paste(g, patch, int(loc[0]), int(loc[1]), transparent=bg)
                return True
        return False

    if bool(np.random.rand() < 0.6):
        for idx in anchor_order:
            loc = _attached_location(g, patch, anchors[idx], bg)
            if loc is not None:
                paste(g, patch, int(loc[0]), int(loc[1]), transparent=bg)
                return True

    try:
        r0, c0 = random_free_location(g, patch, transparent=bg, border=1, padding=1, padding_connectivity=4)
        paste(g, patch, int(r0), int(c0), transparent=bg)
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    H = int(np.random.randint(30, 41))
    W = int(np.random.randint(30, 41))
    g = np.full((H, W), bg, dtype=int)

    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(1, 5))
    n_valid = n_frames + n_solids
    n_distractors = int(np.random.randint(1, min(4, 10 - n_valid)))

    sizes = _sample_sizes(n_valid)
    colors = list(np.random.permutation(np.arange(1, 10)).astype(int))
    valid_colors = colors[:n_valid]
    distractor_colors = colors[n_valid : n_valid + n_distractors]

    specs: list[dict] = []
    for i in range(n_frames):
        specs.append({"kind": "frame", "size": int(sizes[i]), "color": int(valid_colors[i])})
    for i in range(n_solids):
        idx = n_frames + i
        specs.append({"kind": "solid", "size": int(sizes[idx]), "color": int(valid_colors[idx])})
    np.random.shuffle(specs)

    objects: list[dict] = []
    anchors: list[tuple[int, int, int, int]] = []
    try:
        for spec in specs:
            size = int(spec["size"])
            color = int(spec["color"])
            sprite = _make_frame(size, color, bg) if spec["kind"] == "frame" else _make_solid(size, color)
            reserve = np.full((size, size), int(color), dtype=int)
            r0, c0 = random_free_location(
                g,
                reserve,
                transparent=bg,
                border=5,
                padding=2,
                padding_connectivity=4,
            )
            paste(g, sprite, int(r0), int(c0), transparent=bg)
            objects.append(
                {
                    "kind": spec["kind"],
                    "size": size,
                    "r0": int(r0),
                    "c0": int(c0),
                    "sprite": sprite,
                }
            )
            anchors.append((int(r0), int(c0), int(r0 + size - 1), int(c0 + size - 1)))

        for i, dcol in enumerate(distractor_colors):
            shape = DISTRACTOR_SHAPES[int(np.random.randint(0, len(DISTRACTOR_SHAPES)))]
            patch = np.full(shape, int(dcol), dtype=int)
            if not _place_distractor(g, patch, bg, anchors, force_attach=(i == 0)):
                return None

        frame_rows = [int(obj["r0"]) for obj in objects if obj["kind"] == "frame"]
        solid_rows = [int(obj["r0"]) for obj in objects if obj["kind"] == "solid"]
        if len(frame_rows) != len(set(frame_rows)) or len(solid_rows) != len(set(solid_rows)):
            return None

        expected = _assemble_output(objects, bg)
        actual = main(g)
        if not _same_output(actual, expected):
            return None

        if _same_output(expected, _size_only_baseline(g)):
            return None
        if _same_output(expected, _connectivity_only_baseline(g)):
            return None

        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
