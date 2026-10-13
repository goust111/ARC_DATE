from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WORK_SIZE = 30
MAX_GENERATION_ATTEMPTS = 400
QUADRANTS = ("tl", "tr", "bl", "br")
NON_BG_COLORS = tuple(range(1, 10))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _border_set(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pts.add((int(r0), int(c)))
        pts.add((int(r1), int(c)))
    for r in range(r0 + 1, r1):
        pts.add((int(r), int(c0)))
        pts.add((int(r), int(c1)))
    return pts


def _frame_from_component(coords: np.ndarray, color: int) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 5 or w < 5:
        return None
    pts = {(int(r), int(c)) for r, c in coords}
    border = _border_set(r0, c0, r1, c1)
    if pts != border:
        return None
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "inner_h": int(h - 2),
        "inner_w": int(w - 2),
        "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
    }


def _patch_centroid(patch: np.ndarray) -> tuple[float, float]:
    coords = np.argwhere(patch)
    return float(coords[:, 0].mean()), float(coords[:, 1].mean())


def _piece_from_coords(coords: np.ndarray, color: int) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = True
    return {
        "color": int(color),
        "coords": np.array(coords, dtype=int),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "patch": patch,
        "area": int(len(coords)),
        "centroid": (float(coords[:, 0].mean()), float(coords[:, 1].mean())),
        "centroid_offset": _patch_centroid(patch),
    }


def _classify_objects(g: np.ndarray) -> tuple[dict[int, dict[str, object]], dict[int, list[dict[str, object]]]]:
    frames: dict[int, dict[str, object]] = {}
    pieces: dict[int, list[dict[str, object]]] = {}
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        frame: dict[str, object] | None = None
        color_pieces: list[dict[str, object]] = []
        for comp in _components(g == color):
            maybe_frame = _frame_from_component(comp, color)
            if maybe_frame is not None:
                assert frame is None, "expected one frame per color"
                frame = maybe_frame
            else:
                color_pieces.append(_piece_from_coords(comp, color))
        assert frame is not None, "expected one matching frame for every fragment color"
        assert color_pieces, "expected at least one fragment per frame color"
        frames[color] = frame
        pieces[color] = color_pieces
    return frames, pieces


def _quadrant(centroid: tuple[float, float], center: tuple[float, float]) -> str:
    upper = centroid[0] < center[0]
    left = centroid[1] < center[1]
    if upper and left:
        return "tl"
    if upper and not left:
        return "tr"
    if not upper and left:
        return "bl"
    return "br"


def _piece_sort_key(piece: dict[str, object]) -> tuple[object, ...]:
    patch = np.asarray(piece["patch"], dtype=bool)
    r0, c0, _r1, _c1 = piece.get("bbox", (0, 0, 0, 0))
    return (
        -int(piece["area"]),
        int(r0),
        int(c0),
        int(patch.shape[0]),
        int(patch.shape[1]),
        patch.tobytes(),
    )


def _corner_key(r: int, c: int, ph: int, pw: int, ih: int, iw: int, quad: str) -> tuple[int, int, int, int, int, int]:
    if quad[0] == "t":
        dv = int(r)
    else:
        dv = int(ih - ph - r)
    if quad[1] == "l":
        dh = int(c)
    else:
        dh = int(iw - pw - c)
    return (max(dv, dh), dv + dh, dv, dh, int(r), int(c))


def _best_anchor(occupied: np.ndarray, patch: np.ndarray, quad: str) -> tuple[int, int] | None:
    ih, iw = occupied.shape
    ph, pw = patch.shape
    best_key: tuple[int, int, int, int, int, int] | None = None
    best_pos: tuple[int, int] | None = None
    for r in range(ih - ph + 1):
        for c in range(iw - pw + 1):
            if bool((occupied[r : r + ph, c : c + pw] & patch).any()):
                continue
            key = _corner_key(r, c, ph, pw, ih, iw, quad)
            if best_key is None or key < best_key:
                best_key = key
                best_pos = (int(r), int(c))
    return best_pos


def _pack_pieces(inner_h: int, inner_w: int, pieces: list[dict[str, object]]) -> dict[int, tuple[int, int]] | None:
    occupied = np.zeros((inner_h, inner_w), dtype=bool)
    placements: dict[int, tuple[int, int]] = {}
    for quad in QUADRANTS:
        group = [piece for piece in pieces if piece["quadrant"] == quad]
        group.sort(key=_piece_sort_key)
        for piece in group:
            patch = np.asarray(piece["patch"], dtype=bool)
            anchor = _best_anchor(occupied, patch, quad)
            if anchor is None:
                return None
            r0, c0 = anchor
            ph, pw = patch.shape
            occupied[r0 : r0 + ph, c0 : c0 + pw] |= patch
            placements[int(piece["id"])] = (int(r0), int(c0))
    return placements


def _draw_frame(grid: np.ndarray, r0: int, c0: int, outer_h: int, outer_w: int, color: int) -> None:
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> None:
    ph, pw = patch.shape
    window = grid[r0 : r0 + ph, c0 : c0 + pw]
    window[patch] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort same-colored fragments into their matching hollow frames and tuck them into the corresponding interior corners.

    Input structure:
    - The background is black.
    - Several colors each form exactly one hollow rectangular frame with a black interior.
    - Additional small 4-connected monochrome fragments of those same colors are scattered outside the frames.
    - Every fragment color matches one frame color, and fragments do not overlap frames or each other.

    Rule:
    - Match every fragment to the unique frame of the same color.
    - Compare the fragment centroid with the matched frame center to determine one of four quadrants: top-left, top-right, bottom-left, or bottom-right.
    - Erase the scattered fragments from the canvas, then redraw each fragment unchanged inside the matched frame, packed from the corresponding corner inward.
    - When several same-color fragments map to the same corner, place them by decreasing area, choosing the tightest available position nearest that corner.

    Key insight:
    - Color decides which frame absorbs a fragment, while the fragment's original position relative to that frame decides which interior corner it occupies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames, pieces_by_color = _classify_objects(g)
    out = g.copy()

    for color_pieces in pieces_by_color.values():
        for piece in color_pieces:
            coords = np.asarray(piece["coords"], dtype=int)
            out[coords[:, 0], coords[:, 1]] = BG

    for color, frame in frames.items():
        center = frame["center"]
        staged: list[dict[str, object]] = []
        for idx, piece in enumerate(pieces_by_color[color]):
            staged.append(
                {
                    **piece,
                    "id": int(idx),
                    "quadrant": _quadrant(piece["centroid"], center),
                }
            )
        placements = _pack_pieces(int(frame["inner_h"]), int(frame["inner_w"]), staged)
        assert placements is not None, "matched fragments must fit inside their frame"

        r0, c0, _r1, _c1 = frame["bbox"]
        inner_r0 = int(r0 + 1)
        inner_c0 = int(c0 + 1)
        for piece in staged:
            pr, pc = placements[int(piece["id"])]
            _paint_patch(out, np.asarray(piece["patch"], dtype=bool), inner_r0 + pr, inner_c0 + pc, int(color))

    return out


def _random_piece_patch(area: int, rng: random.Random) -> np.ndarray:
    dims = [(h, w) for h in range(1, 5) for w in range(1, 5) if h * w >= area]
    for _ in range(80):
        h, w = rng.choice(dims)
        seed = (rng.randrange(h), rng.randrange(w))
        cells = {seed}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(frontier[rng.randrange(len(frontier))])
        if len(cells) != area:
            continue
        rs = [r for r, _c in cells]
        cs = [c for _r, c in cells]
        r0 = min(rs)
        c0 = min(cs)
        patch = np.zeros((max(rs) - r0 + 1, max(cs) - c0 + 1), dtype=bool)
        for r, c in cells:
            patch[r - r0, c - c0] = True
        rot = rng.randrange(4)
        patch = np.rot90(patch, rot)
        if rng.random() < 0.5:
            patch = np.fliplr(patch)
        if rng.random() < 0.5:
            patch = np.flipud(patch)
        return patch.copy()
    raise RuntimeError("failed to create a connected fragment patch")


def _make_frame_spec(color: int, rng: random.Random) -> dict[str, object] | None:
    for _ in range(120):
        n_pieces = rng.randint(1, 4)
        quad_assignments = [rng.choice(QUADRANTS) for _ in range(n_pieces)]

        pieces: list[dict[str, object]] = []
        next_id = 0
        for quad in QUADRANTS:
            count = quad_assignments.count(quad)
            if count == 0:
                continue
            areas = rng.sample([1, 2, 3, 4], count)
            rng.shuffle(areas)
            for area in areas:
                patch = _random_piece_patch(area, rng)
                pieces.append(
                    {
                        "id": int(next_id),
                        "color": int(color),
                        "patch": patch,
                        "area": int(area),
                        "quadrant": quad,
                        "bbox": (0, 0, patch.shape[0] - 1, patch.shape[1] - 1),
                        "centroid_offset": _patch_centroid(patch),
                    }
                )
                next_id += 1

        max_piece_h = max(int(piece["patch"].shape[0]) for piece in pieces)
        max_piece_w = max(int(piece["patch"].shape[1]) for piece in pieces)
        inner_h = rng.randint(max(5, max_piece_h + 1), 10)
        inner_w = rng.randint(max(5, max_piece_w + 1), 10)
        placements = _pack_pieces(inner_h, inner_w, pieces)
        if placements is None:
            continue
        return {
            "color": int(color),
            "inner_h": int(inner_h),
            "inner_w": int(inner_w),
            "outer_h": int(inner_h + 2),
            "outer_w": int(inner_w + 2),
            "pieces": pieces,
            "placements": placements,
        }
    return None


def _rects_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return not (
        a[2] + pad < b[0] or
        b[2] + pad < a[0] or
        a[3] + pad < b[1] or
        b[3] + pad < a[1]
    )


def _place_frames(frame_specs: list[dict[str, object]], rng: random.Random) -> bool:
    placed: list[tuple[int, int, int, int]] = []
    order = sorted(frame_specs, key=lambda spec: int(spec["outer_h"]) * int(spec["outer_w"]), reverse=True)
    for spec in order:
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        candidates: list[tuple[int, int]] = []
        for r0 in range(WORK_SIZE - outer_h + 1):
            for c0 in range(WORK_SIZE - outer_w + 1):
                box = (int(r0), int(c0), int(r0 + outer_h - 1), int(c0 + outer_w - 1))
                if all(not _rects_too_close(box, other, pad=1) for other in placed):
                    candidates.append((int(r0), int(c0)))
        if not candidates:
            return False
        r0, c0 = candidates[rng.randrange(len(candidates))]
        spec["bbox"] = (int(r0), int(c0), int(r0 + outer_h - 1), int(c0 + outer_w - 1))
        spec["center"] = ((2 * r0 + outer_h - 1) / 2.0, (2 * c0 + outer_w - 1) / 2.0)
        placed.append(spec["bbox"])
    return True


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dr, dc in NEIGHBORS8:
        src_r0 = max(0, -dr)
        src_r1 = mask.shape[0] - max(0, dr)
        src_c0 = max(0, -dc)
        src_c1 = mask.shape[1] - max(0, dc)
        dst_r0 = max(0, dr)
        dst_r1 = dst_r0 + (src_r1 - src_r0)
        dst_c0 = max(0, dc)
        dst_c1 = dst_c0 + (src_c1 - src_c0)
        out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _patch_fits_with_padding(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    ph, pw = patch.shape
    dilated = _dilate8(occupied)
    return not bool((dilated[r0 : r0 + ph, c0 : c0 + pw] & patch).any())


def _bbox_intersects_frame_boxes(
    bbox: tuple[int, int, int, int], frame_boxes: list[tuple[int, int, int, int]]
) -> bool:
    for box in frame_boxes:
        if not (bbox[2] < box[0] or box[2] < bbox[0] or bbox[3] < box[1] or box[3] < bbox[1]):
            return True
    return False


def _matches_quadrant(point: tuple[float, float], center: tuple[float, float], quad: str) -> bool:
    if quad[0] == "t" and not (point[0] < center[0]):
        return False
    if quad[0] == "b" and not (point[0] > center[0]):
        return False
    if quad[1] == "l" and not (point[1] < center[1]):
        return False
    if quad[1] == "r" and not (point[1] > center[1]):
        return False
    return True


def _scatter_pieces(input_grid: np.ndarray, frame_specs: list[dict[str, object]], rng: random.Random) -> bool:
    occupied = input_grid != BG
    frame_boxes = [tuple(map(int, spec["bbox"])) for spec in frame_specs]
    all_pieces: list[tuple[dict[str, object], dict[str, object]]] = []
    for spec in frame_specs:
        for piece in spec["pieces"]:
            all_pieces.append((spec, piece))
    rng.shuffle(all_pieces)
    all_pieces.sort(key=lambda item: int(item[1]["area"]), reverse=True)

    for spec, piece in all_pieces:
        patch = np.asarray(piece["patch"], dtype=bool)
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(WORK_SIZE - ph + 1):
            for c0 in range(WORK_SIZE - pw + 1):
                bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
                if _bbox_intersects_frame_boxes(bbox, frame_boxes):
                    continue
                if not _patch_fits_with_padding(occupied, patch, r0, c0):
                    continue
                centroid_offset = piece["centroid_offset"]
                centroid = (r0 + centroid_offset[0], c0 + centroid_offset[1])
                if not _matches_quadrant(centroid, spec["center"], str(piece["quadrant"])):
                    continue
                candidates.append((int(r0), int(c0)))
        if not candidates:
            return False
        r0, c0 = candidates[rng.randrange(len(candidates))]
        piece["input_bbox"] = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
        centroid_offset = piece["centroid_offset"]
        piece["input_centroid"] = (r0 + centroid_offset[0], c0 + centroid_offset[1])
        _paint_patch(input_grid, patch, r0, c0, int(piece["color"]))
        occupied[r0 : r0 + ph, c0 : c0 + pw] |= patch
    return True


def _render_pair(frame_specs: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    input_grid = np.zeros((WORK_SIZE, WORK_SIZE), dtype=int)
    output_grid = np.zeros_like(input_grid)
    for spec in frame_specs:
        r0, c0, _r1, _c1 = spec["bbox"]
        _draw_frame(input_grid, int(r0), int(c0), int(spec["outer_h"]), int(spec["outer_w"]), int(spec["color"]))
        _draw_frame(output_grid, int(r0), int(c0), int(spec["outer_h"]), int(spec["outer_w"]), int(spec["color"]))
        inner_r0 = int(r0 + 1)
        inner_c0 = int(c0 + 1)
        for piece in spec["pieces"]:
            pr, pc = spec["placements"][int(piece["id"])]
            _paint_patch(output_grid, np.asarray(piece["patch"], dtype=bool), inner_r0 + pr, inner_c0 + pc, int(spec["color"]))
    return input_grid, output_grid


def _crop_with_margin(inp: np.ndarray, out: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BG)
    assert len(coords) > 0
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())

    top = rng.randint(0, r0)
    left = rng.randint(0, c0)
    bottom = rng.randint(r1, inp.shape[0] - 1)
    right = rng.randint(c1, inp.shape[1] - 1)
    return (
        inp[top : bottom + 1, left : right + 1].copy(),
        out[top : bottom + 1, left : right + 1].copy(),
    )


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.randint(2, 3)
    colors = rng.sample(list(NON_BG_COLORS), n_frames)
    frame_specs: list[dict[str, object]] = []
    for color in colors:
        spec = _make_frame_spec(int(color), rng)
        if spec is None:
            return None
        frame_specs.append(spec)

    if not _place_frames(frame_specs, rng):
        return None

    input_grid, output_grid = _render_pair(frame_specs)
    if not _scatter_pieces(input_grid, frame_specs, rng):
        return None

    cropped_inp, cropped_out = _crop_with_margin(input_grid, output_grid, rng)
    if not np.array_equal(main(cropped_inp), cropped_out):
        return None
    return cropped_inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GENERATION_ATTEMPTS} attempts")
