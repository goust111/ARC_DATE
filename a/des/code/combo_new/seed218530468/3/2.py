from __future__ import annotations

import random
from typing import Iterable

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _crop_component(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int, mask: np.ndarray | None = None) -> None:
    ph, pw = patch.shape
    view = dst[top : top + ph, left : left + pw]
    if mask is None:
        view[:, :] = patch
    else:
        view[mask] = patch[mask]


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def _gap_run(side: np.ndarray) -> tuple[int, int] | None | str:
    missing = np.flatnonzero(~side)
    if missing.size == 0:
        return None
    if missing[-1] - missing[0] + 1 != missing.size:
        return "invalid"
    if not 1 <= int(missing.size) <= 3:
        return "invalid"
    return int(missing[0]), int(missing[-1])


def _analyze_container(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> dict | None:
    h, w = mask.shape
    if h < 4 or w < 4:
        return None
    if np.any(mask[1:-1, 1:-1]):
        return None
    if not bool(mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1]):
        return None

    side_arrays = {
        "top": mask[0, 1:-1],
        "bottom": mask[-1, 1:-1],
        "left": mask[1:-1, 0],
        "right": mask[1:-1, -1],
    }
    gap_side = None
    gap_span = None
    for side, arr in side_arrays.items():
        gap = _gap_run(arr)
        if gap == "invalid":
            return None
        if gap is not None:
            if gap_side is not None:
                return None
            gap_side = side
            gap_span = gap
    if gap_side is None or gap_span is None:
        return None

    gap_len = int(gap_span[1] - gap_span[0] + 1)
    expected_area = 2 * h + 2 * w - 4 - gap_len
    if int(mask.sum()) != expected_area:
        return None

    r0, c0, r1, c1 = bbox
    if gap_side in ("top", "bottom"):
        gap_cells = np.arange(1 + gap_span[0], 2 + gap_span[1])
        gap_center = float(c0 + gap_cells.mean())
    else:
        gap_cells = np.arange(1 + gap_span[0], 2 + gap_span[1])
        gap_center = float(r0 + gap_cells.mean())

    return {
        "bbox": (r0, c0, r1, c1),
        "side": gap_side,
        "gap_span": gap_span,
        "gap_len": gap_len,
        "gap_center": gap_center,
    }


def _rotate_for_side(mask: np.ndarray, side: str) -> np.ndarray:
    h, w = mask.shape
    if h == w:
        return mask.copy()
    need_horizontal = side in ("top", "bottom")
    if need_horizontal and w > h:
        return mask.copy()
    if not need_horizontal and h > w:
        return mask.copy()
    return np.rot90(mask, 1)


def _placement_for_piece(container: dict, piece_mask: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = container["bbox"]
    side = container["side"]
    center = float(container["gap_center"])
    ph, pw = piece_mask.shape

    ir0, ic0 = r0 + 1, c0 + 1
    ir1, ic1 = r1 - 1, c1 - 1
    max_top = ir1 - ph + 1
    max_left = ic1 - pw + 1

    if side == "top":
        top = ir0
        left = _round_half_up(center - (pw - 1) / 2)
        left = min(max(left, ic0), max_left)
    elif side == "bottom":
        top = max_top
        left = _round_half_up(center - (pw - 1) / 2)
        left = min(max(left, ic0), max_left)
    elif side == "left":
        left = ic0
        top = _round_half_up(center - (ph - 1) / 2)
        top = min(max(top, ir0), max_top)
    else:
        left = max_left
        top = _round_half_up(center - (ph - 1) / 2)
        top = min(max(top, ir0), max_top)
    return int(top), int(left)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored fragment back into the matching gapped container and rotate it to match the gap side.

    Input structure:
    - Background is black.
    - Several colors each appear in exactly two disconnected monochrome objects: one is a one-pixel-thick hollow rectangle whose border has one contiguous gap, and the other is a small 4-connected fragment outside every container.
    - The gap lies on exactly one side of the rectangle and spans 1-3 border cells without removing a corner.

    Rule:
    - Pair every fragment with the container of the same color.
    - Detect the container's gap side and the gap center along that side.
    - Rotate the fragment by 0 or 90 degrees so its bounding-box long axis is parallel to the gap side; square bounding boxes are left unchanged.
    - Erase the fragment from its original location, then place it inside the container flush to the gap side, with its bounding-box center aligned to the gap center along the perpendicular axis.
    - Keep each container border and its gap unchanged.

    Key insight:
    - Color gives the fragment↔container pairing, but the container gap independently encodes both the fragment orientation class and the side of the interior where the fragment must be inserted.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    for color in colors:
        comps = _components(g == color)
        container = None
        piece_masks: list[np.ndarray] = []
        piece_boxes: list[tuple[int, int, int, int]] = []
        for coords in comps:
            comp_mask, bbox = _crop_component(coords)
            info = _analyze_container(comp_mask, bbox)
            if info is not None:
                if container is not None:
                    raise AssertionError("expected one container per color")
                container = info
            else:
                piece_masks.append(comp_mask)
                piece_boxes.append(bbox)

        if container is None or len(piece_masks) != 1:
            raise AssertionError("expected one fragment and one container per color")

        piece_mask = piece_masks[0]
        pr0, pc0, pr1, pc1 = piece_boxes[0]
        out[pr0 : pr1 + 1, pc0 : pc1 + 1][piece_mask] = bg

        rotated = _rotate_for_side(piece_mask, container["side"])
        top, left = _placement_for_piece(container, rotated)
        _paste(out, _mask_to_patch(rotated, color), top, left, mask=rotated)

    return out


def _pick_gap_start(edge_len: int, gap_len: int, piece_span: int) -> int | None:
    starts = list(range(0, edge_len - gap_len + 1))
    if not starts:
        return None

    mid = (edge_len - 1) / 2
    categorized: dict[str, list[int]] = {"low": [], "mid": [], "high": []}
    for start in starts:
        center = start + (gap_len - 1) / 2
        left = _round_half_up(1 + center - (piece_span - 1) / 2)
        min_left = 1
        max_left = edge_len - piece_span + 1
        if not (min_left <= left <= max_left):
            continue
        if center < mid - 0.5:
            categorized["low"].append(start)
        elif center > mid + 0.5:
            categorized["high"].append(start)
        else:
            categorized["mid"].append(start)

    available = [bucket for bucket, vals in categorized.items() if vals]
    if not available:
        return None
    bucket = random.choice(available)
    return int(random.choice(categorized[bucket]))


def _container_patch(
    h: int,
    w: int,
    color: int,
    side: str,
    gap_len: int,
    gap_start: int,
) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if side == "top":
        patch[0, 1 + gap_start : 1 + gap_start + gap_len] = BG
    elif side == "bottom":
        patch[-1, 1 + gap_start : 1 + gap_start + gap_len] = BG
    elif side == "left":
        patch[1 + gap_start : 1 + gap_start + gap_len, 0] = BG
    else:
        patch[1 + gap_start : 1 + gap_start + gap_len, -1] = BG
    return patch


def _orbit_mask(h: int, w: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            rr, cc = h - 1 - r, w - 1 - c
            orbit = {(r, c), (rr, cc)}
            for cell in orbit:
                seen.add(cell)
            orbits.append(sorted(orbit))
    return orbits


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _random_symmetric_mask(h: int, w: int) -> np.ndarray:
    orbits = _orbit_mask(h, w)
    min_area = max(3, (h * w + 2) // 3)
    for _ in range(200):
        mask = np.zeros((h, w), dtype=bool)
        p = random.uniform(0.45, 0.8)
        for orbit in orbits:
            if random.random() < p:
                for r, c in orbit:
                    mask[r, c] = True
        if int(mask.sum()) < min_area:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if h >= 4 and w >= 4 and not mask[1:-1, 1:-1].any() and not bool(mask.all()):
            continue
        if not _is_connected(mask):
            continue
        return mask
    return np.ones((h, w), dtype=bool)


def _make_piece(ih: int, iw: int, side: str) -> tuple[np.ndarray, np.ndarray]:
    need_horizontal = side in ("top", "bottom")
    square_bbox = random.random() < 0.5

    if square_bbox:
        sizes = [s for s in range(2, min(ih, iw, 5) + 1)]
        if not sizes:
            raise ValueError("no square piece fits")
        size = random.choice(sizes)
        final_mask = _random_symmetric_mask(size, size)
        input_mask = np.rot90(final_mask, random.randint(0, 3))
        return final_mask, input_mask

    dims: list[tuple[int, int]] = []
    if need_horizontal:
        for h in range(2, min(ih, 4) + 1):
            for w in range(h + 1, min(iw, 6) + 1):
                dims.append((h, w))
    else:
        for h in range(3, min(ih, 6) + 1):
            for w in range(2, min(iw, min(h - 1, 4)) + 1):
                dims.append((h, w))
    if not dims:
        raise ValueError("no non-square piece fits")

    h, w = random.choice(dims)
    final_mask = _random_symmetric_mask(h, w)
    input_mask = final_mask if random.random() < 0.5 else np.rot90(final_mask, 1)
    return final_mask, input_mask


def _find_slot(occupied: np.ndarray, ph: int, pw: int, pad: int = 1) -> tuple[int, int] | None:
    h, w = occupied.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h - ph + 1):
        for c in range(w - pw + 1):
            r0 = max(0, r - pad)
            c0 = max(0, c - pad)
            r1 = min(h, r + ph + pad)
            c1 = min(w, c + pw + pad)
            if not occupied[r0:r1, c0:c1].any():
                candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _reserve(occupied: np.ndarray, top: int, left: int, ph: int, pw: int, pad: int = 1) -> None:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + ph + pad)
    c1 = min(w, left + pw + pad)
    occupied[r0:r1, c0:c1] = True


def _expected_output(grid: np.ndarray, specs: Iterable[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for spec in specs:
        fr, fc = spec["fragment_pos"]
        in_mask = spec["input_piece"]
        out[fr : fr + in_mask.shape[0], fc : fc + in_mask.shape[1]][in_mask] = BG
        top, left = spec["placed_pos"]
        final_mask = spec["final_piece"]
        _paste(out, _mask_to_patch(final_mask, spec["color"]), top, left, mask=final_mask)
    return out


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.randint(2, 4)
    colors = random.sample(list(range(1, 10)), n_pairs)
    specs: list[dict] = []

    for color in colors:
        made = False
        for _ in range(60):
            side = random.choice(["top", "bottom", "left", "right"])
            oh = random.randint(6, 9)
            ow = random.randint(6, 9)
            ih, iw = oh - 2, ow - 2
            try:
                final_piece, input_piece = _make_piece(ih, iw, side)
            except ValueError:
                continue

            gap_axis = ow - 2 if side in ("top", "bottom") else oh - 2
            gap_len = random.randint(1, min(3, gap_axis))
            piece_span = final_piece.shape[1] if side in ("top", "bottom") else final_piece.shape[0]
            gap_start = _pick_gap_start(gap_axis, gap_len, piece_span)
            if gap_start is None:
                continue

            container_patch = _container_patch(oh, ow, color, side, gap_len, gap_start)
            container_info = _analyze_container(container_patch == color, (0, 0, oh - 1, ow - 1))
            if container_info is None:
                continue
            placed_pos = _placement_for_piece(container_info, final_piece)
            top, left = placed_pos
            if top < 1 or left < 1 or top + final_piece.shape[0] > oh - 1 or left + final_piece.shape[1] > ow - 1:
                continue

            specs.append(
                {
                    "color": color,
                    "side": side,
                    "container_patch": container_patch,
                    "container_info": container_info,
                    "final_piece": final_piece,
                    "input_piece": input_piece,
                    "placed_pos": placed_pos,
                }
            )
            made = True
            break
        if not made:
            return None

    canvas_h = random.randint(22, 30)
    canvas_w = random.randint(22, 30)
    grid = np.zeros((canvas_h, canvas_w), dtype=int)
    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)

    place_order = sorted(
        range(len(specs)),
        key=lambda i: -(specs[i]["container_patch"].shape[0] * specs[i]["container_patch"].shape[1]),
    )
    for idx in place_order:
        spec = specs[idx]
        patch = spec["container_patch"]
        pos = _find_slot(occupied, patch.shape[0], patch.shape[1], pad=1)
        if pos is None:
            return None
        top, left = pos
        spec["container_pos"] = pos
        _paste(grid, patch, top, left)
        _reserve(occupied, top, left, patch.shape[0], patch.shape[1], pad=1)

    frag_order = sorted(
        range(len(specs)),
        key=lambda i: -(specs[i]["input_piece"].shape[0] * specs[i]["input_piece"].shape[1]),
    )
    for idx in frag_order:
        spec = specs[idx]
        patch = _mask_to_patch(spec["input_piece"], spec["color"])
        pos = _find_slot(occupied, patch.shape[0], patch.shape[1], pad=1)
        if pos is None:
            return None
        top, left = pos
        spec["fragment_pos"] = pos
        _paste(grid, patch, top, left, mask=spec["input_piece"])
        _reserve(occupied, top, left, patch.shape[0], patch.shape[1], pad=1)

    for spec in specs:
        cr, cc = spec["container_pos"]
        local_container = spec["container_info"]
        global_container = dict(local_container)
        r0, c0, r1, c1 = local_container["bbox"]
        global_container["bbox"] = (cr + r0, cc + c0, cr + r1, cc + c1)
        global_container["gap_center"] = float(
            local_container["gap_center"] + (cc if local_container["side"] in ("top", "bottom") else cr)
        )
        spec["placed_pos"] = _placement_for_piece(global_container, spec["final_piece"])

    expected = _expected_output(grid, specs)
    if _get_bg(grid) != BG:
        return None
    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GEN_ATTEMPTS} attempts")
