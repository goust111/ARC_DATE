import numpy as np
from collections import deque


BG = 0
YELLOW = 4
COLOR_POOL = [1, 2, 3, 5, 6, 7, 8, 9]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    out = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out, (int(r0), int(c0))


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _assemble_crops(items: list[tuple[tuple[int, int, int, int], np.ndarray]]) -> np.ndarray:
    items = sorted(items, key=lambda item: (item[0][1], item[0][0]))
    height = max(int(crop.shape[0]) for _, crop in items)
    width = sum(int(crop.shape[1]) for _, crop in items) + len(items) - 1
    out = np.full((height, width), BG, dtype=int)
    col = 0
    for _, crop in items:
        h, w = crop.shape
        out[:h, col : col + w] = crop
        col += w + 1
    return out


def _is_frame_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 6 or w < 6:
        return False
    full_perimeter = 2 * (h + w) - 4
    if len(coords) >= full_perimeter or len(coords) < full_perimeter - 10:
        return False
    side_hits = [False, False, False, False]
    for r, c in coords:
        if r == r0:
            side_hits[0] = True
        if r == r1:
            side_hits[1] = True
        if c == c0:
            side_hits[2] = True
        if c == c1:
            side_hits[3] = True
        if r not in (r0, r1) and c not in (c0, c1):
            return False
    return all(side_hits)


def _frame_gaps(grid: np.ndarray, bbox_: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = bbox_
    gaps = []
    for side in ("top", "bottom"):
        r = r0 if side == "top" else r1
        start = None
        for c in range(c0 + 1, c1):
            if grid[r, c] == BG:
                if start is None:
                    start = c
            elif start is not None:
                cells = [(r, cc) for cc in range(start, c)]
                gaps.append(
                    {
                        "side": side,
                        "cells": cells,
                        "cellset": set(cells),
                        "bbox": (r, start, r, c - 1),
                        "mask": np.ones((1, c - start), dtype=bool),
                    }
                )
                start = None
        if start is not None:
            cells = [(r, cc) for cc in range(start, c1)]
            gaps.append(
                {
                    "side": side,
                    "cells": cells,
                    "cellset": set(cells),
                    "bbox": (r, start, r, c1 - 1),
                    "mask": np.ones((1, c1 - start), dtype=bool),
                }
            )
    for side in ("left", "right"):
        c = c0 if side == "left" else c1
        start = None
        for r in range(r0 + 1, r1):
            if grid[r, c] == BG:
                if start is None:
                    start = r
            elif start is not None:
                cells = [(rr, c) for rr in range(start, r)]
                gaps.append(
                    {
                        "side": side,
                        "cells": cells,
                        "cellset": set(cells),
                        "bbox": (start, c, r - 1, c),
                        "mask": np.ones((r - start, 1), dtype=bool),
                    }
                )
                start = None
        if start is not None:
            cells = [(rr, c) for rr in range(start, r1)]
            gaps.append(
                {
                    "side": side,
                    "cells": cells,
                    "cellset": set(cells),
                    "bbox": (start, c, r1 - 1, c),
                    "mask": np.ones((r1 - start, 1), dtype=bool),
                }
            )
    return gaps


def _extract_piece_patch(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
    colors = sorted({int(grid[r, c]) for r, c in coords})
    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "patch": patch,
        "colors": colors,
    }


def _match_piece(frame: dict, gap_list: list[dict], piece: dict) -> dict:
    fr0, fc0, fr1, fc1 = frame["bbox"]
    owner_color = frame["color"]
    other_colors = [color for color in piece["colors"] if color != owner_color]
    assert len(other_colors) == 1
    content_color = int(other_colors[0])
    candidates = []
    for gap_idx, gap in enumerate(gap_list):
        for rot in range(4):
            rot_patch = np.rot90(piece["patch"], rot)
            owner_mask = rot_patch == owner_color
            if not owner_mask.any():
                continue
            norm_mask, (or0, oc0) = _crop_mask(owner_mask)
            if norm_mask.shape != gap["mask"].shape or not np.array_equal(norm_mask, gap["mask"]):
                continue
            dr = gap["bbox"][0] - or0
            dc = gap["bbox"][1] - oc0
            content_cells = []
            ok = True
            for pr, pc in np.argwhere(rot_patch != BG):
                gr = int(pr + dr)
                gc = int(pc + dc)
                if not (fr0 <= gr <= fr1 and fc0 <= gc <= fc1):
                    ok = False
                    break
                color = int(rot_patch[pr, pc])
                if color == owner_color:
                    if (gr, gc) not in gap["cellset"]:
                        ok = False
                        break
                else:
                    if gr in (fr0, fr1) or gc in (fc0, fc1):
                        ok = False
                        break
                    content_cells.append((gr, gc))
            if ok:
                candidates.append(
                    {
                        "gap_idx": gap_idx,
                        "content_color": content_color,
                        "content_cells": content_cells,
                    }
                )
    assert len(candidates) == 1
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BG)
    frames = []
    pieces = []
    for coords in comps:
        colors = sorted({int(g[r, c]) for r, c in coords})
        if len(colors) == 1 and _is_frame_component(coords):
            frames.append(
                {
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "color": int(colors[0]),
                }
            )
        elif len(colors) == 2:
            pieces.append(_extract_piece_patch(g, coords))

    assert 2 <= len(frames) <= 3
    frame_colors = {frame["color"] for frame in frames}
    by_color = {frame["color"]: frame for frame in frames}
    frame_gaps = {frame["color"]: _frame_gaps(g, frame["bbox"]) for frame in frames}
    for gap_list in frame_gaps.values():
        assert 1 <= len(gap_list) <= 2

    matched_by_frame: dict[int, list[dict]] = {frame["color"]: [] for frame in frames}
    used_gaps: dict[int, set[int]] = {frame["color"]: set() for frame in frames}
    for piece in pieces:
        owner_colors = set(piece["colors"]) & frame_colors
        assert len(owner_colors) == 1
        owner_color = int(next(iter(owner_colors)))
        match = _match_piece(by_color[owner_color], frame_gaps[owner_color], piece)
        assert match["gap_idx"] not in used_gaps[owner_color]
        used_gaps[owner_color].add(match["gap_idx"])
        matched_by_frame[owner_color].append(match)

    crops = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        crop = np.full((h, w), BG, dtype=int)
        crop[0, :] = frame["color"]
        crop[-1, :] = frame["color"]
        crop[:, 0] = frame["color"]
        crop[:, -1] = frame["color"]
        for match in matched_by_frame[frame["color"]]:
            for gr, gc in match["content_cells"]:
                lr = gr - r0
                lc = gc - c0
                if crop[lr, lc] == BG:
                    crop[lr, lc] = match["content_color"]
                elif crop[lr, lc] == frame["color"]:
                    raise AssertionError("content landed on the frame border")
                elif crop[lr, lc] != YELLOW:
                    crop[lr, lc] = YELLOW
        crops.append((frame["bbox"], crop))
    return _assemble_crops(crops)


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _pick(rng: np.random.Generator, seq):
    return seq[_randint(rng, 0, len(seq) - 1)]


def _sample_without_replacement(rng: np.random.Generator, seq, k: int):
    idx = rng.choice(len(seq), size=k, replace=False)
    return [seq[int(i)] for i in idx]


def _gap_cells(side: str, h: int, w: int, start: int, length: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, start + i) for i in range(length)]
    if side == "bottom":
        return [(h - 1, start + i) for i in range(length)]
    if side == "left":
        return [(start + i, 0) for i in range(length)]
    return [(start + i, w - 1) for i in range(length)]


def _inward_step(side: str) -> tuple[int, int]:
    return {
        "top": (1, 0),
        "bottom": (-1, 0),
        "left": (0, 1),
        "right": (0, -1),
    }[side]


def _manhattan_path(start: tuple[int, int], goal: tuple[int, int], row_first: bool) -> list[tuple[int, int]]:
    r, c = start
    out = [(r, c)]
    if row_first:
        while r != goal[0]:
            r += 1 if goal[0] > r else -1
            out.append((r, c))
        while c != goal[1]:
            c += 1 if goal[1] > c else -1
            out.append((r, c))
    else:
        while c != goal[1]:
            c += 1 if goal[1] > c else -1
            out.append((r, c))
        while r != goal[0]:
            r += 1 if goal[0] > r else -1
            out.append((r, c))
    return out


def _inside_interior(h: int, w: int, cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 < r < h - 1 and 0 < c < w - 1


def _make_content_shape(
    rng: np.random.Generator,
    h: int,
    w: int,
    gap: dict,
    target: tuple[int, int],
    row_first: bool,
) -> set[tuple[int, int]]:
    owner_cells = gap["cells"]
    anchor = owner_cells[_randint(rng, 0, len(owner_cells) - 1)]
    dr, dc = _inward_step(gap["side"])
    start = (anchor[0] + dr, anchor[1] + dc)
    path = _manhattan_path(start, target, row_first=row_first)
    content = set(path)

    if len(owner_cells) >= 3 and rng.random() < 0.5:
        alt = owner_cells[len(owner_cells) // 2]
        extra = (alt[0] + dr, alt[1] + dc)
        if _inside_interior(h, w, extra):
            content.add(extra)

    if rng.random() < 0.8:
        branch_base = _pick(rng, path[max(0, len(path) // 2 - 1) : len(path)])
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        rng.shuffle(dirs)
        branch_len = _pick(rng, [1, 2])
        for bdr, bdc in dirs:
            cells = []
            cr, cc = branch_base
            for _ in range(branch_len):
                cr += bdr
                cc += bdc
                if not _inside_interior(h, w, (cr, cc)):
                    cells = []
                    break
                cells.append((cr, cc))
            if cells:
                content.update(cells)
                break
    return content


def _coords_to_patch(owner_cells, content_cells, owner_color: int, content_color: int) -> np.ndarray:
    all_cells = owner_cells + list(content_cells)
    rows = [r for r, _ in all_cells]
    cols = [c for _, c in all_cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in owner_cells:
        patch[r - r0, c - c0] = owner_color
    for r, c in content_cells:
        patch[r - r0, c - c0] = content_color
    return patch


def _expected_frame_crop(frame: dict) -> np.ndarray:
    crop = np.full((frame["h"], frame["w"]), BG, dtype=int)
    crop[0, :] = frame["color"]
    crop[-1, :] = frame["color"]
    crop[:, 0] = frame["color"]
    crop[:, -1] = frame["color"]
    for piece in frame["pieces"]:
        for r, c in piece["content_cells"]:
            if crop[r, c] == BG:
                crop[r, c] = piece["content_color"]
            elif crop[r, c] != YELLOW:
                crop[r, c] = YELLOW
    return crop


def _frame_mask_connected(h: int, w: int, gaps: list[dict]) -> bool:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    for gap in gaps:
        for r, c in gap["cells"]:
            mask[r, c] = False
    comps = _components(mask)
    return len(comps) == 1


def _choose_gap_count_pattern(rng: np.random.Generator, frame_count: int) -> list[int]:
    if frame_count == 2:
        return _pick(rng, [[1, 1], [1, 2], [2, 1], [2, 2]])
    patterns = [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2], [2, 2, 1], [2, 1, 2], [1, 2, 2]]
    return list(_pick(rng, patterns))


def _random_target(rng: np.random.Generator, h: int, w: int) -> tuple[int, int]:
    return (_randint(rng, 2, h - 3), _randint(rng, 2, w - 3))


def _build_frame_pieces(rng: np.random.Generator, frame: dict) -> bool:
    gap_count = frame["gap_count"]
    h, w = frame["h"], frame["w"]
    if gap_count == 1:
        sides = [_pick(rng, ["top", "bottom", "left", "right"])]
    else:
        sides = list(
            _pick(
                rng,
                [
                    ("top", "left"),
                    ("top", "right"),
                    ("bottom", "left"),
                    ("bottom", "right"),
                ],
            )
        )
    used_lengths: set[int] = set()
    gaps = []
    for side in sides:
        side_span = w if side in ("top", "bottom") else h
        lengths = [length for length in range(2, min(4, side_span - 3) + 1) if length not in used_lengths]
        if not lengths:
            return False
        length = _pick(rng, lengths)
        start = _randint(rng, 1, side_span - length - 1)
        gaps.append(
            {
                "side": side,
                "length": length,
                "cells": _gap_cells(side, h, w, start, length),
            }
        )
        used_lengths.add(length)
    if not _frame_mask_connected(h, w, gaps):
        return False

    overlap = gap_count == 2 and rng.random() < 0.5
    targets = []
    if gap_count == 1:
        targets = [_random_target(rng, h, w)]
    elif overlap:
        shared = _random_target(rng, h, w)
        targets = [shared, shared]
    else:
        for _ in range(40):
            t0 = _random_target(rng, h, w)
            t1 = _random_target(rng, h, w)
            if abs(t0[0] - t1[0]) + abs(t0[1] - t1[1]) >= 4:
                targets = [t0, t1]
                break
        if not targets:
            return False

    pieces = []
    existing: set[tuple[int, int]] = set()
    row_first_choices = [True, False]
    for idx, gap in enumerate(gaps):
        success = False
        for _ in range(40):
            row_first = row_first_choices[idx % 2] if overlap else bool(_randint(rng, 0, 1))
            content = _make_content_shape(rng, h, w, gap, targets[idx], row_first=row_first)
            if not content:
                continue
            inter = existing & content
            if overlap and idx == 1:
                if not inter:
                    continue
            elif inter:
                continue
            success = True
            pieces.append(
                {
                    "owner_cells": list(gap["cells"]),
                    "content_cells": sorted(content),
                }
            )
            existing |= content
            break
        if not success:
            return False

    frame["gaps"] = gaps
    frame["pieces"] = pieces
    return True


def _paint_input_frame(grid: np.ndarray, frame: dict):
    r0, c0, r1, c1 = frame["bbox"]
    grid[r0, c0 : c1 + 1] = frame["color"]
    grid[r1, c0 : c1 + 1] = frame["color"]
    grid[r0 : r1 + 1, c0] = frame["color"]
    grid[r0 : r1 + 1, c1] = frame["color"]
    for gap in frame["gaps"]:
        for lr, lc in gap["cells"]:
            grid[r0 + lr, c0 + lc] = BG


def _place_patch(
    rng: np.random.Generator,
    grid: np.ndarray,
    patch: np.ndarray,
    forbidden: list[tuple[int, int, int, int]],
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    for _ in range(200):
        r0 = _randint(rng, 1, h - ph - 2)
        c0 = _randint(rng, 1, w - pw - 2)
        bbox_ = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        if any(_rect_overlap(bbox_, rect, margin=1) for rect in forbidden):
            continue
        region = grid[r0 : r0 + ph, c0 : c0 + pw]
        mask = patch != BG
        if np.any(region[mask] != BG):
            continue
        region[mask] = patch[mask]
        forbidden.append(bbox_)
        return True
    return False


def _generate_scene(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    frame_count = _pick(rng, [2, 3])
    gap_counts = _choose_gap_count_pattern(rng, frame_count)
    total_pieces = sum(gap_counts)
    frame_colors = _sample_without_replacement(rng, COLOR_POOL, frame_count)
    content_colors = _sample_without_replacement(rng, [c for c in COLOR_POOL if c not in frame_colors], total_pieces)

    canvas_h = _randint(rng, 28, 36)
    canvas_w = _randint(rng, 32, 42)
    frames = []
    for idx in range(frame_count):
        placed = False
        for _ in range(200):
            fh = _randint(rng, 7, 11)
            fw = _randint(rng, 7, 12)
            r0 = _randint(rng, 1, canvas_h - fh - 2)
            c0 = _randint(rng, 1, canvas_w - fw - 2)
            bbox_ = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if any(_rect_overlap(bbox_, other["bbox"], margin=3) for other in frames):
                continue
            frame = {
                "bbox": bbox_,
                "h": fh,
                "w": fw,
                "color": int(frame_colors[idx]),
                "gap_count": int(gap_counts[idx]),
            }
            if not _build_frame_pieces(rng, frame):
                continue
            frames.append(frame)
            placed = True
            break
        if not placed:
            return None

    color_idx = 0
    for frame in frames:
        for piece in frame["pieces"]:
            piece["content_color"] = int(content_colors[color_idx])
            color_idx += 1

    input_grid = np.full((canvas_h, canvas_w), BG, dtype=int)
    for frame in frames:
        _paint_input_frame(input_grid, frame)

    forbidden = [frame["bbox"] for frame in frames]
    for frame in frames:
        for piece in frame["pieces"]:
            patch = _coords_to_patch(piece["owner_cells"], piece["content_cells"], frame["color"], piece["content_color"])
            patch = np.rot90(patch, _randint(rng, 0, 3))
            if not _place_patch(rng, input_grid, patch, forbidden):
                return None

    crops = []
    for frame in frames:
        crops.append((frame["bbox"], _expected_frame_crop(frame)))
    expected = _assemble_crops(crops)
    return input_grid, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        scene = _generate_scene(rng)
        if scene is None:
            continue
        grid, expected = scene
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
