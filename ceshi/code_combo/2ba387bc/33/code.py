# Combo variant of 2ba387bc (index 33)
# combo_spec: D13/2+D15/2
# Name: 空间分离陷阱+排序标记

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for combo D13/2 + D15/2
=======================================

1. Concrete behavior and trace
------------------------------
- Original task:
  - Input contains scattered 4x4 objects on background.
  - Hollow frames and solid blocks are both classified by shape.
  - Within each type, output order is the input row order.
- Fused variant:
  - Classification stays shape-based: frame objects go to the left output column,
    solid objects go to the right output column.
  - But the input is spatially biased: all frames are placed in the upper region,
    all solids in the lower region. This creates a position-based trap.
  - Row order is no longer the sorting rule. Each 4x4 object carries a visible
    1-pixel marker at its top-left corner; the marker color encodes priority.
  - The output stacks each type by marker priority, not by input row.
- Small trace:
  - Suppose three upper frames appear from top to bottom as:
    yellow-marked, red-marked, green-marked.
  - Original logic would output them in that same top-to-bottom order.
  - New logic ignores those rows and instead uses the fixed visible priority
    `red < green < yellow`, so the output order becomes:
    red-marked, green-marked, yellow-marked.
  - The same rule applies independently to the lower solid group.
- Main difference:
  - D13/2 makes the spatial split visually salient.
  - D15/2 then breaks the natural but wrong hypothesis that "top-to-bottom input
    position determines stack order".

2. Complexity calibration
-------------------------
- Only three marker colors are used, repeated across every sample. This keeps the
  symbolic rule human-visible instead of requiring a large arbitrary color order.
- Every sample contains exactly three frames and three solids, so all three marker
  priorities are present twice per sample. That gives enough repeated evidence for
  humans to infer the marker ordering from 2-4 train pairs.
- Object body colors are disjoint from marker colors. This prevents ambiguity
  about which pixel is the marker.
- Marker placement is always the same corner (top-left). Varying both corner and
  color would add unnecessary degrees of freedom and make the task visually noisy.

3. Original-solver assumption audit
-----------------------------------
- `bg = get_bg(g)`: still valid. Background remains the most common color.
- `find_components(... monochromatic=False)`: still valid and required, because a
  marked object may contain marker color + body color (+ background interior for
  frames) but is still one connected 4x4 component.
- `bbox_h(b) == 4 and bbox_w(b) == 4`: still valid. Sortable objects remain 4x4.
- Old frame test relied on the full border being one color. That is no longer
  valid because the top-left marker deliberately breaks border uniformity.
- New frame test uses only the 2x2 center:
    frame iff `sub[1:3, 1:3]` is all background.
  This matches the fused design because markers are only on a corner pixel.
- Old sorting by `(row, col)` is invalid. The fused rule sorts by marker priority.
- Output geometry stays the same: 8 columns wide, 4 rows per object.

4. Placement legality
---------------------
- Frames are placed only in a top sub-canvas; solids only in a bottom sub-canvas.
  A blank separator row remains between those regions.
- Placement uses `random_free_location(... padding=1, connectivity=4)` inside each
  sub-canvas, so objects neither overlap nor 4-neighbor-touch.
- Since every patch is 4x4 and placement is border-constrained, no object crosses
  the grid boundary.

5. Visual observability and cleanliness
---------------------------------------
- The spatial trap is obvious at a glance: frames are high, solids are low.
- The sorting cue is also obvious: each object has one conspicuous corner pixel
  whose color is reused across samples.
- Because the marker is embedded inside the 4x4 footprint rather than floating
  outside it, the scene stays tidy and component extraction stays unambiguous.
- No human-invisible property is used. The priority is a visible categorical color
  relation repeated across the training distribution.

6. Non-degeneration metric
--------------------------
- Metric: compare two outputs on the same generated input:
  1. fused rule = shape classification + marker-priority sorting
  2. original rule = shape classification + row sorting
- If the two outputs are identical, the sample is degenerate because the marker
  code had no behavioral effect.
- The generator explicitly rejects such samples and only returns grids where the
  fused output differs from the row-sorted baseline.

7. New solver assumptions and generator guarantees
--------------------------------------------------
- Assumption: each valid object has its marker at `sub[0, 0]`.
  Guarantee: generator always stamps the marker at the top-left corner.
- Assumption: marker colors come from a small fixed task-specific palette with a
  stable visible order.
  Guarantee: generator uses exactly `{2, 3, 4}` with priority `2 < 3 < 4`.
- Assumption: every valid object is either a marked frame or a marked solid.
  Guarantee: generator builds only those two shapes.
- Assumption: objects stay spatially disconnected.
  Guarantee: generator uses padding-based placement in separated sub-canvases.
"""


MARKER_PRIORITY_COLORS: tuple[int, ...] = (2, 3, 4)
MARKER_RANK = {col: idx for idx, col in enumerate(MARKER_PRIORITY_COLORS)}
BODY_COLORS: tuple[int, ...] = (1, 5, 6, 7, 8, 9)


def _make_frame(body_col: int, marker_col: int, bg: int = 0) -> np.ndarray:
    patch = np.full((4, 4), int(body_col), dtype=int)
    patch[1:3, 1:3] = int(bg)
    patch[0, 0] = int(marker_col)
    return patch


def _make_solid(body_col: int, marker_col: int) -> np.ndarray:
    patch = np.full((4, 4), int(body_col), dtype=int)
    patch[0, 0] = int(marker_col)
    return patch


def _is_frame(sub: np.ndarray, bg: int) -> bool:
    return bool((sub[1:3, 1:3] == int(bg)).all())


def _collect_objects(grid: np.ndarray) -> tuple[int, list[tuple[str, int, int, int, np.ndarray]]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    objects: list[tuple[str, int, int, int, np.ndarray]] = []
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        marker_col = int(sub[0, 0])
        marker_rank = int(MARKER_RANK.get(marker_col, len(MARKER_PRIORITY_COLORS)))
        kind = "frame" if _is_frame(sub, bg) else "solid"
        objects.append((kind, marker_rank, int(r0), int(c0), sub))
    return bg, objects


def _stack_objects(
    bg: int,
    objects: list[tuple[str, int, int, int, np.ndarray]],
    *,
    by_marker: bool,
) -> np.ndarray:
    frames = [obj for obj in objects if obj[0] == "frame"]
    solids = [obj for obj in objects if obj[0] == "solid"]

    if by_marker:
        frames.sort(key=lambda t: (int(t[1]), int(t[2]), int(t[3])))
        solids.sort(key=lambda t: (int(t[1]), int(t[2]), int(t[3])))
    else:
        frames.sort(key=lambda t: (int(t[2]), int(t[3])))
        solids.sort(key=lambda t: (int(t[2]), int(t[3])))

    out_h = 4 * max(len(frames), len(solids))
    out = np.full((out_h, 8), int(bg), dtype=int)

    for i, (_kind, _rank, _r0, _c0, sub) in enumerate(frames):
        paste(out, sub, i * 4, 0)
    for i, (_kind, _rank, _r0, _c0, sub) in enumerate(solids):
        paste(out, sub, i * 4, 4)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Detect every 4x4 marked object.
    - Classify by shape only:
      - frame if the center 2x2 is background
      - solid otherwise
    - Ignore the input row order.
    - Sort each type by the top-left marker color priority `2 < 3 < 4`.
    - Stack frames in the left output column and solids in the right column.
    """
    bg, objects = _collect_objects(grid)
    return _stack_objects(bg, objects, by_marker=True)


def _nonidentity_marker_assignment(n: int) -> list[int]:
    base = list(MARKER_PRIORITY_COLORS[:n])
    if n <= 1:
        return base
    shift = int(np.random.randint(1, n))
    return base[shift:] + base[:shift]


def _place_group(
    region: np.ndarray,
    row_offset: int,
    kind: str,
    body_cols: list[int],
    bg: int,
) -> list[tuple[str, int, int, int, np.ndarray]]:
    placed: list[tuple[str, int, int, int, np.ndarray]] = []

    for body_col in body_cols:
        if kind == "frame":
            base_patch = _make_frame(body_col, body_col, bg)
        else:
            base_patch = _make_solid(body_col, body_col)
        r0_local, c0 = random_free_location(
            region,
            base_patch,
            transparent=bg,
            border=1,
            padding=1,
            padding_connectivity=4,
        )
        paste(region, base_patch, r0_local, c0, transparent=bg)
        placed.append((kind, -1, row_offset + int(r0_local), int(c0), base_patch.copy()))

    placed.sort(key=lambda t: (int(t[2]), int(t[3])))
    assigned_markers = _nonidentity_marker_assignment(len(placed))

    updated: list[tuple[str, int, int, int, np.ndarray]] = []
    for (kind0, _rank0, r0, c0, patch), marker_col in zip(placed, assigned_markers):
        region[int(r0 - row_offset), int(c0)] = int(marker_col)
        patch[0, 0] = int(marker_col)
        updated.append((kind0, int(MARKER_RANK[int(marker_col)]), int(r0), int(c0), patch))
    return updated


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    bg, objects = _collect_objects(grid)
    fused = _stack_objects(bg, objects, by_marker=True)
    row_sorted = _stack_objects(bg, objects, by_marker=False)

    frame_rows = [int(obj[2]) for obj in objects if obj[0] == "frame"]
    solid_rows = [int(obj[2]) for obj in objects if obj[0] == "solid"]

    return {
        "n_frames": len(frame_rows),
        "n_solids": len(solid_rows),
        "separated": int(max(frame_rows) + 3 < min(solid_rows)),
        "diff_from_row_sort": int(not np.array_equal(fused, row_sorted)),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    H = int(np.random.randint(24, 31))
    W = int(np.random.randint(22, 31))
    g = np.full((H, W), int(bg), dtype=int)

    mid = H // 2
    top_region = g[: mid - 1, :]
    bottom_region = g[mid + 1 :, :]
    if top_region.shape[0] < 7 or bottom_region.shape[0] < 7:
        return None

    body_cols = list(np.random.permutation(BODY_COLORS))
    frame_body_cols = [int(c) for c in body_cols[:3]]
    solid_body_cols = [int(c) for c in body_cols[3:6]]

    try:
        objects: list[tuple[str, int, int, int, np.ndarray]] = []
        objects.extend(_place_group(top_region, 0, "frame", frame_body_cols, bg))
        objects.extend(_place_group(bottom_region, mid + 1, "solid", solid_body_cols, bg))

        metrics = _behavior_metrics(g)
        if metrics["n_frames"] != 3 or metrics["n_solids"] != 3:
            return None
        if metrics["separated"] != 1 or metrics["diff_from_row_sort"] != 1:
            return None

        expected = _stack_objects(bg, objects, by_marker=True)
        if not np.array_equal(main(g), expected):
            return None
        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 300 attempts")
