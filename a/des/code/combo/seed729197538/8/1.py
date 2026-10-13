from __future__ import annotations

import random
from typing import Iterable

import numpy as np


FRAME_TRANSFORM = {
    1: "v",   # blue
    2: "h",   # red
    3: "id",  # green
}

FRAME_COLORS = [1, 2, 3]
COLOR_POOL = list(range(10))
GAP_SHAPES = ["line2", "line3", "corner3"]
MAX_GENERATION_ATTEMPTS = 400


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _shape_key(mask: np.ndarray) -> tuple:
    variants = []
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.append((int(rot.shape[0]), int(rot.shape[1]), rot.tobytes()))
    return min(variants)


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (int(rot.shape[0]), int(rot.shape[1]), rot.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(rot.copy())
    return out


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
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _perimeter_cycle(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = [(r0, c) for c in range(c0, c1 + 1)]
    cells.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    cells.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
    cells.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return cells


def _cyclic_missing_run(present: list[bool]) -> list[int] | None:
    n = len(present)
    if n == 0 or all(present) or not any(present):
        return None
    starts = [i for i in range(n) if (not present[i]) and present[(i - 1) % n]]
    if len(starts) != 1:
        return None
    start = int(starts[0])
    run = []
    i = start
    while not present[i]:
        run.append(i)
        i = (i + 1) % n
        if i == start:
            break
    if len(run) == n:
        return None
    if any((not present[j]) and (j not in run) for j in range(n)):
        return None
    return run


def _frame_patch(
    height: int,
    width: int,
    frame_color: int,
    piece_color: int,
    gap_cells: Iterable[tuple[int, int]],
    transform: str,
    bg: int,
) -> np.ndarray:
    patch = np.full((height, width), bg, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    for r, c in gap_cells:
        patch[int(r), int(c)] = piece_color
    if transform == "h":
        patch = np.fliplr(patch)
    elif transform == "v":
        patch = np.flipud(patch)
    else:
        assert transform == "id"
    return patch


def _analyze_frame(g: np.ndarray, bg: int, color: int) -> dict | None:
    coords = np.argwhere(g == color)
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 4 or width < 4:
        return None

    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    if np.any((sub != bg) & (sub != color)):
        return None

    for r, c in coords:
        if r not in (r0, r1) and c not in (c0, c1):
            return None

    cycle = _perimeter_cycle(r0, c0, r1, c1)
    present = [int(g[r, c]) == color for r, c in cycle]
    missing_idx = _cyclic_missing_run(present)
    if missing_idx is None or len(missing_idx) not in (1, 2, 3):
        return None

    gap_abs = [cycle[i] for i in missing_idx]
    if any(int(g[r, c]) != bg for r, c in gap_abs):
        return None

    expected_count = len(cycle) - len(gap_abs)
    if int(coords.shape[0]) != expected_count:
        return None
    if np.any(sub[1:-1, 1:-1] != bg):
        return None

    gap_rel = np.array([(r - r0, c - c0) for r, c in gap_abs], dtype=int)
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap_rel": gap_rel,
        "gap_key": _shape_key(_tight_mask(gap_rel)),
    }


def _find_frames(g: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(set(map(int, g.ravel())) - {int(bg)}):
        info = _analyze_frame(g, int(bg), int(color))
        if info is not None:
            assert int(color) in FRAME_TRANSFORM
            frames.append(info)
    assert frames, "expected at least one frame"
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _find_pieces(g: np.ndarray, bg: int, frames: list[dict]) -> list[dict]:
    frame_pixels = np.zeros(g.shape, dtype=bool)
    for frame in frames:
        frame_pixels |= g == int(frame["color"])

    comps = _components((g != bg) & (~frame_pixels))
    pieces: list[dict] = []
    for comp in comps:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "piece must be monochrome"
        color = colors.pop()
        mask = _tight_mask(comp)
        pieces.append(
            {
                "color": int(color),
                "mask": mask,
                "key": _shape_key(mask),
            }
        )
    assert pieces, "expected outside candidate pieces"
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each frame gap with the outside piece of the same shape (up to rotation),
    fill the gap, then flip the completed frame in-place according to its border color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    frames = _find_frames(g, bg)
    pieces = _find_pieces(g, bg, frames)

    by_key: dict[tuple, dict] = {}
    for piece in pieces:
        assert piece["key"] not in by_key, "piece shapes must be uniquely matchable"
        by_key[piece["key"]] = piece

    out = np.full_like(g, bg)
    for frame in frames:
        key = frame["gap_key"]
        assert key in by_key, "every frame gap must have a matching outside piece"
        piece = by_key.pop(key)
        r0, c0, r1, c1 = frame["bbox"]
        patch = _frame_patch(
            r1 - r0 + 1,
            c1 - c0 + 1,
            int(frame["color"]),
            int(piece["color"]),
            frame["gap_rel"],
            FRAME_TRANSFORM[int(frame["color"])],
            int(bg),
        )
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    assert not by_key, "unused pieces remain"
    return out


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _sample_gap(shape: str, height: int, width: int, transform: str) -> np.ndarray:
    for _ in range(200):
        if shape.startswith("line"):
            length = 2 if shape == "line2" else 3
            side = random.choice(["top", "bottom", "left", "right"])
            if side in {"top", "bottom"}:
                if width < length + 2:
                    continue
                start = random.randint(1, width - length - 1)
                row = 0 if side == "top" else height - 1
                cells = [(row, start + i) for i in range(length)]
            else:
                if height < length + 2:
                    continue
                start = random.randint(1, height - length - 1)
                col = 0 if side == "left" else width - 1
                cells = [(start + i, col) for i in range(length)]
        else:
            corner = random.choice(["tl", "tr", "bl", "br"])
            if corner == "tl":
                cells = [(0, 0), (0, 1), (1, 0)]
            elif corner == "tr":
                cells = [(0, width - 1), (0, width - 2), (1, width - 1)]
            elif corner == "bl":
                cells = [(height - 1, 0), (height - 2, 0), (height - 1, 1)]
            else:
                cells = [(height - 1, width - 1), (height - 2, width - 1), (height - 1, width - 2)]

        gap = np.array(cells, dtype=int)
        if transform != "id":
            moved = _transform_rel(gap, height, width, transform)
            if {tuple(x) for x in moved.tolist()} == {tuple(x) for x in gap.tolist()}:
                continue
        return gap
    raise RuntimeError("failed to sample a non-degenerate gap")


def _transform_rel(coords: np.ndarray, height: int, width: int, transform: str) -> np.ndarray:
    out = coords.copy()
    if transform == "h":
        out[:, 1] = width - 1 - out[:, 1]
    elif transform == "v":
        out[:, 0] = height - 1 - out[:, 0]
    else:
        assert transform == "id"
    return out


def _draw_incomplete_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int, gap: np.ndarray, bg: int) -> None:
    grid[top, left : left + width] = color
    grid[top + height - 1, left : left + width] = color
    grid[top : top + height, left] = color
    grid[top : top + height, left + width - 1] = color
    for r, c in gap:
        grid[top + int(r), left + int(c)] = bg


def _can_place_piece(
    occupied: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    frame_boxes: list[tuple[int, int, int, int]],
) -> bool:
    h, w = mask.shape
    H, W = occupied.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False

    piece_box = (top, left, top + h - 1, left + w - 1)
    for frame_box in frame_boxes:
        if _boxes_intersect(piece_box, frame_box, pad=1):
            return False

    cells = np.argwhere(mask)
    for dr, dc in cells:
        r = top + int(dr)
        c = left + int(dc)
        r0 = max(0, r - 1)
        c0 = max(0, c - 1)
        r1 = min(H, r + 2)
        c1 = min(W, c + 2)
        if np.any(occupied[r0:r1, c0:c1]):
            return False
    return True


def _place_piece(
    grid: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    color: int,
    frame_boxes: list[tuple[int, int, int, int]],
) -> bool:
    H, W = grid.shape
    positions = [(r, c) for r in range(H - mask.shape[0] + 1) for c in range(W - mask.shape[1] + 1)]
    random.shuffle(positions)
    for top, left in positions:
        if not _can_place_piece(occupied, mask, int(top), int(left), frame_boxes):
            continue
        cells = np.argwhere(mask)
        for dr, dc in cells:
            rr = int(top + dr)
            cc = int(left + dc)
            grid[rr, cc] = color
            occupied[rr, cc] = True
        return True
    return False


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    bg_choices = [c for c in COLOR_POOL if c not in FRAME_COLORS]
    bg = random.choice(bg_choices)
    n_frames = random.randint(2, 3)

    frame_colors = random.sample(FRAME_COLORS, n_frames)
    piece_colors = random.sample([c for c in COLOR_POOL if c not in FRAME_COLORS and c != bg], n_frames)
    shape_choices = random.sample(GAP_SHAPES, n_frames)

    H = random.randint(18, 28)
    W = random.randint(18, 28)
    frame_specs = []
    frame_boxes: list[tuple[int, int, int, int]] = []

    for frame_color, piece_color, shape in zip(frame_colors, piece_colors, shape_choices, strict=False):
        placed = False
        transform = FRAME_TRANSFORM[int(frame_color)]
        for _ in range(200):
            height = random.randint(6, 10)
            width = random.randint(6, 10)
            if height >= H - 2 or width >= W - 2:
                continue
            top = random.randint(1, H - height - 1)
            left = random.randint(1, W - width - 1)
            box = (top, left, top + height - 1, left + width - 1)
            if any(_boxes_intersect(box, other, pad=2) for other in frame_boxes):
                continue
            gap = _sample_gap(shape, height, width, transform)
            frame_specs.append(
                {
                    "frame_color": int(frame_color),
                    "piece_color": int(piece_color),
                    "shape": shape,
                    "top": int(top),
                    "left": int(left),
                    "height": int(height),
                    "width": int(width),
                    "gap": gap,
                }
            )
            frame_boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    inp = np.full((H, W), bg, dtype=int)
    out = np.full((H, W), bg, dtype=int)

    occupied = np.zeros((H, W), dtype=bool)
    for spec, box in zip(frame_specs, frame_boxes, strict=False):
        _draw_incomplete_frame(
            inp,
            spec["top"],
            spec["left"],
            spec["height"],
            spec["width"],
            spec["frame_color"],
            spec["gap"],
            bg,
        )
        frame_patch = _frame_patch(
            spec["height"],
            spec["width"],
            spec["frame_color"],
            spec["piece_color"],
            spec["gap"],
            FRAME_TRANSFORM[spec["frame_color"]],
            bg,
        )
        out[spec["top"] : spec["top"] + spec["height"], spec["left"] : spec["left"] + spec["width"]] = frame_patch
        rr0, cc0, rr1, cc1 = box
        occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] |= inp[rr0 : rr1 + 1, cc0 : cc1 + 1] != bg

    for spec in frame_specs:
        base_mask = _tight_mask(spec["gap"])
        piece_mask = random.choice(_unique_rotations(base_mask))
        if not _place_piece(inp, occupied, piece_mask, spec["piece_color"], frame_boxes):
            return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_example()
        if built is None:
            continue
        inp, out = built
        try:
            if np.array_equal(main(inp), out):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate valid input")
