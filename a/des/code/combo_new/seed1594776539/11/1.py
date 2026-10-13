import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
BASE_SIZE = 30
GEN_RETRIES = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(pts)
    return parts


def bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def tight_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def rotate_mask_and_point(mask: np.ndarray, point: tuple[int, int], k: int) -> tuple[np.ndarray, tuple[int, int]]:
    rot_mask = np.rot90(mask, k)
    marker = np.zeros(mask.shape, dtype=bool)
    marker[point] = True
    rot_marker = np.rot90(marker, k)
    pr, pc = np.argwhere(rot_marker)[0]
    return rot_mask, (int(pr), int(pc))


def dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(h, h - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(w, w - dc)
            dst_r0 = max(0, dr)
            dst_r1 = min(h, h + dr)
            dst_c0 = max(0, dc)
            dst_c1 = min(w, w + dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each external fragment back into its color-marked cavity inside the grey frame.

    Input structure:
    - The background is black.
    - A single grey rectangular frame encloses several grey internal walls, creating 2 to 4 disjoint cavities.
    - Each cavity contains exactly one non-black marker pixel on its inner wall, and marker colors are unique.
    - Outside the frame, there is one connected colored fragment per cavity; each fragment is monochrome except for one interface pixel whose color matches a cavity marker.

    Rule:
    - Identify the central grey frame and extract each enclosed cavity as a connected non-grey region inside the frame.
    - Pair every external fragment with the cavity whose marker color matches the fragment's singleton interface color.
    - Rotate the fragment by 0, 90, 180, or 270 degrees until its occupied shape matches the cavity exactly and its interface pixel lands on the cavity marker.
    - Fill the cavity with the fragment's body color, keep the marker color at the aligned interface cell, and erase the fragment from its original outside position.

    Key insight:
    - The interface color gives the fragment-to-cavity pairing, while the interface pixel position breaks rotational ambiguity and determines the unique placement.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    grey_parts = components(g == GREY)
    assert grey_parts, "expected a grey frame"
    frame_coords = max(grey_parts, key=len)
    frame_r0, frame_c0, frame_r1, frame_c1 = bbox_from_coords(frame_coords)

    inner = g[frame_r0 + 1 : frame_r1, frame_c0 + 1 : frame_c1]
    cavity_parts = components(inner != GREY)
    cavities: dict[int, dict] = {}
    for part in cavity_parts:
        abs_coords = [(frame_r0 + 1 + r, frame_c0 + 1 + c) for r, c in part]
        marker_cells = [(r, c) for r, c in abs_coords if int(g[r, c]) not in (BLACK, GREY)]
        assert len(marker_cells) == 1, "each cavity must have exactly one marker"
        marker_r, marker_c = marker_cells[0]
        marker_color = int(g[marker_r, marker_c])
        mask, bbox = tight_mask(abs_coords)
        cavities[marker_color] = {
            "mask": mask,
            "bbox": bbox,
            "marker_pos": (marker_r - bbox[0], marker_c - bbox[1]),
            "marker_abs": (marker_r, marker_c),
        }

    color_parts = components((g != BLACK) & (g != GREY))
    pieces: dict[int, dict] = {}
    piece_pixels: list[tuple[int, int]] = []
    for part in color_parts:
        if all(frame_r0 < r < frame_r1 and frame_c0 < c < frame_c1 for r, c in part):
            continue
        counts = Counter(int(g[r, c]) for r, c in part)
        assert len(counts) == 2, "each piece should have body color plus one interface color"
        body_color, body_count = counts.most_common(1)[0]
        interface_colors = [color for color, count in counts.items() if color != body_color and count == 1]
        assert body_count >= 2 and len(interface_colors) == 1, "piece interface must be a singleton color"
        interface_color = interface_colors[0]
        iface_cell = next((r, c) for r, c in part if int(g[r, c]) == interface_color)
        mask, bbox = tight_mask(part)
        pieces[interface_color] = {
            "mask": mask,
            "bbox": bbox,
            "iface_pos": (iface_cell[0] - bbox[0], iface_cell[1] - bbox[1]),
            "body_color": int(body_color),
        }
        piece_pixels.extend(part)

    assert set(cavities) == set(pieces), "marker colors must pair cavities with pieces one-to-one"

    out = g.copy()
    for r, c in piece_pixels:
        out[r, c] = BLACK

    for marker_color, cavity in cavities.items():
        piece = pieces[marker_color]
        matches = 0
        for k in range(4):
            rot_mask, rot_point = rotate_mask_and_point(piece["mask"], piece["iface_pos"], k)
            if rot_mask.shape == cavity["mask"].shape and np.array_equal(rot_mask, cavity["mask"]) and rot_point == cavity["marker_pos"]:
                matches += 1
        assert matches == 1, "expected a unique rotation per piece"

        r0, c0, _r1, _c1 = cavity["bbox"]
        out[r0 : r0 + cavity["mask"].shape[0], c0 : c0 + cavity["mask"].shape[1]][cavity["mask"]] = piece["body_color"]
        mr, mc = cavity["marker_abs"]
        out[mr, mc] = marker_color

    return out


def _random_partition(ch: int, cw: int, regions: int, rng: random.Random) -> np.ndarray | None:
    cells = [(r, c) for r in range(ch) for c in range(cw)]
    seeds = rng.sample(cells, regions)
    labels = -np.ones((ch, cw), dtype=int)
    frontiers = [set() for _ in range(regions)]

    for idx, (r, c) in enumerate(seeds):
        labels[r, c] = idx
    for idx, (r, c) in enumerate(seeds):
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < ch and 0 <= nc < cw and labels[nr, nc] == -1:
                frontiers[idx].add((nr, nc))

    remaining = ch * cw - regions
    sizes = [1] * regions
    while remaining > 0:
        choices = [idx for idx in range(regions) if frontiers[idx]]
        if not choices:
            return None
        weights = [1.0 / sizes[idx] for idx in choices]
        chosen = rng.choices(choices, weights=weights, k=1)[0]
        cell = rng.choice(tuple(frontiers[chosen]))
        cr, cc = cell
        if labels[cr, cc] != -1:
            frontiers[chosen].discard(cell)
            continue
        labels[cr, cc] = chosen
        sizes[chosen] += 1
        remaining -= 1
        for frontier in frontiers:
            frontier.discard(cell)
        for nr, nc in neighbors4(cr, cc):
            if 0 <= nr < ch and 0 <= nc < cw and labels[nr, nc] == -1:
                frontiers[chosen].add((nr, nc))
    return labels


def _labels_to_fine(labels: np.ndarray) -> np.ndarray:
    ch, cw = labels.shape
    fine = -np.ones((3 * ch - 1, 3 * cw - 1), dtype=int)

    for r in range(ch):
        for c in range(cw):
            label = int(labels[r, c])
            rr = 3 * r
            cc = 3 * c
            fine[rr : rr + 2, cc : cc + 2] = label
            if c + 1 < cw and int(labels[r, c + 1]) == label:
                fine[rr : rr + 2, cc + 2] = label
            if r + 1 < ch and int(labels[r + 1, c]) == label:
                fine[rr + 2, cc : cc + 2] = label

    return fine


def _marker_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    picks: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for nr, nc in neighbors4(r, c):
                if nr < 0 or nr >= h or nc < 0 or nc >= w or not mask[nr, nc]:
                    picks.append((r, c))
                    break
    return picks


def _symmetry_count(mask: np.ndarray, marker: tuple[int, int]) -> int:
    total = 0
    for k in range(4):
        rot_mask, rot_marker = rotate_mask_and_point(mask, marker, k)
        if rot_mask.shape == mask.shape and np.array_equal(rot_mask, mask) and rot_marker == marker:
            total += 1
    return total


def _build_layout(rng: random.Random) -> tuple[np.ndarray, list[dict]] | None:
    regions = rng.choice([2, 3, 4])
    ch = rng.choice([3, 4])
    cw = rng.choice([3, 4])
    if ch * cw < regions + 1:
        return None

    labels = _random_partition(ch, cw, regions, rng)
    if labels is None:
        return None
    fine_labels = _labels_to_fine(labels)

    masks: list[np.ndarray] = []
    markers: list[tuple[int, int]] = []
    for label in range(regions):
        mask = fine_labels == label
        if mask.sum() < 4:
            return None
        candidates = _marker_candidates(mask)
        rng.shuffle(candidates)
        marker = None
        for cand in candidates:
            if _symmetry_count(mask, cand) == 1:
                marker = cand
                break
        if marker is None:
            return None
        masks.append(mask)
        markers.append(marker)

    if not any(mask.shape[0] != mask.shape[1] or _symmetry_count(mask, marker) == 1 for mask, marker in zip(masks, markers)):
        return None

    frame = np.full((fine_labels.shape[0] + 2, fine_labels.shape[1] + 2), GREY, dtype=int)
    frame[1:-1, 1:-1] = np.where(fine_labels >= 0, BLACK, GREY)

    color_order = rng.sample(COLOR_POOL, 2 * regions)
    marker_colors = color_order[:regions]
    body_colors = color_order[regions:]
    holes: list[dict] = []
    for idx in range(regions):
        mask = masks[idx]
        coords = [(r, c) for r in range(mask.shape[0]) for c in range(mask.shape[1]) if mask[r, c]]
        tight, bbox = tight_mask(coords)
        rel_marker = (markers[idx][0] - bbox[0], markers[idx][1] - bbox[1])
        fr, fc = markers[idx]
        frame[1 + fr, 1 + fc] = marker_colors[idx]
        holes.append(
            {
                "mask": tight,
                "marker_pos": rel_marker,
                "marker_color": marker_colors[idx],
                "body_color": body_colors[idx],
            }
        )

    return frame, holes


def _patch_from_hole(mask: np.ndarray, marker_pos: tuple[int, int], marker_color: int, body_color: int, rotation: int) -> np.ndarray:
    rot_mask, rot_marker = rotate_mask_and_point(mask, marker_pos, rotation)
    patch = np.zeros(rot_mask.shape, dtype=int)
    patch[rot_mask] = body_color
    patch[rot_marker] = marker_color
    return patch


def _can_place(patch: np.ndarray, grid: np.ndarray, forbidden: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False
    mask = patch != BLACK
    return not np.any(forbidden[r0 : r0 + h, c0 : c0 + w] & mask)


def _random_crop(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    coords = np.argwhere(grid != BLACK)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = rng.randint(0, min(3, int(r0)))
    left = rng.randint(0, min(3, int(c0)))
    bottom = rng.randint(0, min(3, int(grid.shape[0] - 1 - r1)))
    right = rng.randint(0, min(3, int(grid.shape[1] - 1 - c1)))
    return grid[int(r0) - top : int(r1) + bottom + 1, int(c0) - left : int(c1) + right + 1]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    built = _build_layout(rng)
    if built is None:
        return None
    frame, holes = built

    base = np.zeros((BASE_SIZE, BASE_SIZE), dtype=int)
    fh, fw = frame.shape
    frame_r0 = max(1, (BASE_SIZE - fh) // 2 + rng.randint(-2, 2))
    frame_c0 = max(1, (BASE_SIZE - fw) // 2 + rng.randint(-2, 2))
    frame_r0 = min(frame_r0, BASE_SIZE - fh - 1)
    frame_c0 = min(frame_c0, BASE_SIZE - fw - 1)
    base[frame_r0 : frame_r0 + fh, frame_c0 : frame_c0 + fw] = frame

    forbidden = np.zeros_like(base, dtype=bool)
    zone_r0 = max(0, frame_r0 - 1)
    zone_c0 = max(0, frame_c0 - 1)
    zone_r1 = min(BASE_SIZE, frame_r0 + fh + 1)
    zone_c1 = min(BASE_SIZE, frame_c0 + fw + 1)
    forbidden[zone_r0:zone_r1, zone_c0:zone_c1] = True

    pieces = []
    for hole in holes:
        rotation = rng.randrange(4)
        patch = _patch_from_hole(hole["mask"], hole["marker_pos"], hole["marker_color"], hole["body_color"], rotation)
        pieces.append((int(np.count_nonzero(patch)), patch))
    pieces.sort(key=lambda item: -item[0])

    for _area, patch in pieces:
        placed = False
        ph, pw = patch.shape
        for _ in range(300):
            r0 = rng.randint(0, BASE_SIZE - ph)
            c0 = rng.randint(0, BASE_SIZE - pw)
            if not _can_place(patch, base, forbidden, r0, c0):
                continue
            mask = patch != BLACK
            base[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
            piece_zone = np.zeros_like(base, dtype=bool)
            piece_zone[r0 : r0 + ph, c0 : c0 + pw] = mask
            forbidden |= dilate8(piece_zone)
            placed = True
            break
        if not placed:
            return None

    cropped = _random_crop(base, rng)
    if max(cropped.shape) > 30:
        return None

    solved = main(cropped)
    if np.array_equal(solved, cropped):
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
